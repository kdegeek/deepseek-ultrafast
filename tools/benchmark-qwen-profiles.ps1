<#
Serial resident Qwen profile experiments. The JSON input is an array of named
profiles with launcher parameters and optional process-local comparison flags.
Only a process started by this run is stopped, after path/PID/start-time readback.
No build, model download, OS change, or competing model load is performed.
#>
param(
    [Parameter(Mandatory)][string]$Profiles,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [ValidateRange(1,5)][int]$Repeats = 3,
    [ValidateRange(256,1024)][int]$DecodeTokens = 384,
    [ValidateRange(0,2)][double]$BenchmarkTemperature = 0,
    [switch]$Parity,
    [string]$ParityBaseline,
    [switch]$Populated4K,
    [switch]$Extended,
    [switch]$Cancellation,
    [switch]$Acceptance,
    [switch]$Trace
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath "$PSScriptRoot/..").Path
$python = Join-Path $env:USERPROFILE '.unsloth/studio/unsloth_studio/Scripts/python.exe'
$out = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $out) { throw 'OutputDirectory exists; preserve prior evidence with a new directory.' }
[IO.Directory]::CreateDirectory($out) | Out-Null
$matrix = @(Get-Content -LiteralPath $Profiles -Raw | ConvertFrom-Json -AsHashtable)
$names = @{}
foreach ($profile in $matrix) {
    if ($profile.name -notmatch '^[a-zA-Z0-9_-]+$' -or $names.ContainsKey($profile.name)) { throw 'Profile names must be unique simple identifiers.' }
    $names[$profile.name] = $true
    foreach ($key in $profile.Keys) {
        if ($key -notin 'name','parameters','batch_logit_capacity','disable_hip_graphs','always_clear','draft_sampling','target_tokens','fixture_seed') { throw "Unknown profile key: $key" }
    }
    foreach ($key in $profile.parameters.Keys) {
        if ($key -notin 'Backend','Quant','Profile','Context','Batch','UBatch','Threads','KvType','MtpDraft','DFlashDraft','DraftSampling','RequestTimeoutSeconds','BuildVariant') { throw "Unsupported benchmark launch parameter: $key" }
    }
    if ($profile.draft_sampling -and $profile.draft_sampling -notin 'greedy','probabilistic') { throw 'Invalid draft sampling mode.' }
    if ($Extended -and !$profile.target_tokens) { throw 'Extended profiles require target_tokens.' }
}
if ($Cancellation -and !$Extended) { throw 'Cancellation requires a passed extended fixture.' }
if ($Acceptance -and ($Extended -or $Populated4K -or $Trace)) { throw 'Acceptance includes its own bounded benchmark/context checks; choose a separate mode.' }
if ($Trace -and ($Extended -or $Populated4K -or $Parity)) { throw 'Trace diagnostics must run separately from acceptance/benchmarks.' }
$matrix | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$out/profiles.json" -Encoding utf8
$results = [Collections.Generic.List[object]]::new()
$baselineParity = if ($ParityBaseline) { (Resolve-Path -LiteralPath $ParityBaseline).Path } else { $null }
foreach ($profile in $matrix) {
    $started = $null
    $observer = $null
    $memoryPath = "$out/$($profile.name)-memory.json"
    $stopFile = "$out/$($profile.name)-memory.stop"
    $row = [ordered]@{ name=$profile.name; started_utc=[DateTime]::UtcNow.ToString('o'); passed=$false }
    $envCapacity = [Environment]::GetEnvironmentVariable('QWEN_BATCH_LOGIT_CAPACITY','Process')
    $envGraphs = [Environment]::GetEnvironmentVariable('GGML_CUDA_DISABLE_GRAPHS','Process')
    $envClear = [Environment]::GetEnvironmentVariable('QWEN_ALWAYS_CLEAR','Process')
    $envTrace = [Environment]::GetEnvironmentVariable('QWEN_TRACE_TARGET_LOGITS','Process')
    try {
        [Environment]::SetEnvironmentVariable('QWEN_BATCH_LOGIT_CAPACITY',$(if ($profile.batch_logit_capacity) {'1'} else {$null}),'Process')
        [Environment]::SetEnvironmentVariable('GGML_CUDA_DISABLE_GRAPHS',$(if ($profile.disable_hip_graphs) {'1'} else {$null}),'Process')
        [Environment]::SetEnvironmentVariable('QWEN_ALWAYS_CLEAR',$(if ($profile.always_clear) {'1'} else {$null}),'Process')
        [Environment]::SetEnvironmentVariable('QWEN_TRACE_TARGET_LOGITS',$(if ($Trace) {'1'} else {$null}),'Process')
        $parameters = @{ Quant='q8'; Backend='Vulkan'; MtpDraft=0; DFlashDraft=0; DraftSampling='greedy' }
        foreach ($key in $profile.parameters.Keys) { $parameters[$key] = $profile.parameters[$key] }
        if ($profile.draft_sampling) { $parameters.DraftSampling = $profile.draft_sampling }
        Write-Output "Starting serial profile $($profile.name)"
        $launch = (& "$PSScriptRoot/serve-qwen.ps1" @parameters | Out-String)
        $started = $launch | ConvertFrom-Json
        $launch | Set-Content -LiteralPath "$out/$($profile.name)-startup.json" -Encoding utf8
        $row.pid = $started.pid
        $row.identity = $started.health.identity
        $context = $started.health.identity.context
        $observeArgs = @('-NoProfile','-File',"$PSScriptRoot/measure-qwen-memory.ps1",'-ServerPid',"$($started.pid)",'-DurationSeconds','21600','-IntervalSeconds','5','-Output',$memoryPath,'-StopFile',$stopFile)
        $observeArgs = $observeArgs | ForEach-Object { '"' + $_ + '"' }
        $observer = Start-Process -FilePath (Get-Command pwsh).Source -ArgumentList $observeArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out/$($profile.name)-memory.stdout.log" -RedirectStandardError "$out/$($profile.name)-memory.stderr.log"
        if ($Trace) {
            $tracePath = "$out/$($profile.name)-trace.json"
            & $python "$PSScriptRoot/optimize-qwen.py" --suite trace --context $context --decode-tokens $DecodeTokens --output $tracePath
            if ($LASTEXITCODE -ne 0) { throw "Target diagnostic failed: $tracePath" }
            $row.trace = (Get-Content $tracePath -Raw | ConvertFrom-Json).checks
            Get-Content -LiteralPath "$($started.run_directory)/stderr.log" | Where-Object { $_ -match '"type":"target_logits_trace"' } | Set-Content -LiteralPath "$out/$($profile.name)-target-logits.jsonl" -Encoding utf8
        } elseif ($Extended) {
            $longPath = "$out/$($profile.name)-extended.json"
            $fixtureSeed = if ($profile.fixture_seed) { $profile.fixture_seed } else { 20261115 }
            & $python "$PSScriptRoot/optimize-qwen.py" --suite extended --context $context --target-tokens $profile.target_tokens --fixture-seed $fixtureSeed --decode-tokens $DecodeTokens --output $longPath
            $longReport = Get-Content $longPath -Raw | ConvertFrom-Json
            $row.extended = $longReport.extended_summary
            if (!$longReport.passed) { throw "Extended acceptance failed: $longPath" }
            if ($Cancellation) {
                $cancelPath = "$out/$($profile.name)-cancel.json"
                & $python "$PSScriptRoot/optimize-qwen.py" --suite cancel --context $context --cancel-from $longPath --decode-tokens $DecodeTokens --output $cancelPath
                if ($LASTEXITCODE -ne 0) { throw "Extended cancellation failed: $cancelPath" }
                $row.cancellation = (Get-Content $cancelPath -Raw | ConvertFrom-Json).checks
            }
        } elseif ($Acceptance) {
            $acceptancePath = "$out/$($profile.name)-acceptance.json"
            & $python "$PSScriptRoot/validate-qwen.py" --suite all --context $context --repeats $Repeats --decode-tokens $DecodeTokens --benchmark-temperature $BenchmarkTemperature --output $acceptancePath
            $row.acceptance = (Get-Content $acceptancePath -Raw | ConvertFrom-Json).checks
            if ($LASTEXITCODE -ne 0) { throw "Hardware/API acceptance failed: $acceptancePath" }
            $specPath = "$out/$($profile.name)-spec-contract.json"
            & $python "$PSScriptRoot/optimize-qwen.py" --suite spec-contract --context $context --output $specPath
            $row.spec_contract = (Get-Content $specPath -Raw | ConvertFrom-Json).checks
            if ($LASTEXITCODE -ne 0) { throw "Speculative request contracts failed: $specPath" }
        } else {
            $benchPath = "$out/$($profile.name)-bench.json"
            & $python "$PSScriptRoot/validate-qwen.py" --suite bench --context $context --repeats $Repeats --decode-tokens $DecodeTokens --benchmark-temperature $BenchmarkTemperature --output $benchPath
            if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $benchPath" }
            $row.benchmark = (Get-Content $benchPath -Raw | ConvertFrom-Json).benchmark_summary
        }
        if ($Populated4K) {
            $longPath = "$out/$($profile.name)-4k.json"
            & $python "$PSScriptRoot/validate-qwen.py" --suite long-context --context $context --long-targets 4096 --decode-tokens $DecodeTokens --output $longPath
            if ($LASTEXITCODE -ne 0) { throw "Populated 4K check failed: $longPath" }
            $row.populated_4k = (Get-Content $longPath -Raw | ConvertFrom-Json).checks
        }
        if ($Parity) {
            $parityPath = "$out/$($profile.name)-parity.json"
            $compare = @()
            if ($baselineParity) { $compare = @('--compare',$baselineParity) }
            & $python "$PSScriptRoot/optimize-qwen.py" --suite parity --context $context --output $parityPath @compare
            $row.parity_passed = ($LASTEXITCODE -eq 0)
            if (!$baselineParity) { $baselineParity = $parityPath }
            # Preserve numerical/parity failures and continue the bounded matrix;
            # no candidate is selected or published automatically by this runner.
        }
        $row.passed = !$Parity -or $row.parity_passed
    } catch {
        $row.error = $_.Exception.Message
        Write-Warning $row.error
    } finally {
        [Environment]::SetEnvironmentVariable('QWEN_BATCH_LOGIT_CAPACITY',$envCapacity,'Process')
        [Environment]::SetEnvironmentVariable('GGML_CUDA_DISABLE_GRAPHS',$envGraphs,'Process')
        [Environment]::SetEnvironmentVariable('QWEN_ALWAYS_CLEAR',$envClear,'Process')
        [Environment]::SetEnvironmentVariable('QWEN_TRACE_TARGET_LOGITS',$envTrace,'Process')
        if ($observer) {
            [IO.File]::WriteAllText($stopFile,'finished')
            if (!$observer.WaitForExit(15000)) { throw 'Memory observer did not stop within its bounded shutdown window.' }
            if (Test-Path -LiteralPath $memoryPath) {
                $memory = Get-Content -LiteralPath $memoryPath -Raw | ConvertFrom-Json
                $row.memory = $memory.summary
                if ($memory.pressure_observed -or $memory.observer_error) { $row.passed=$false; $row.memory_error=$memory.observer_error }
            }
        }
        if ($started) {
            $process = Get-Process -Id $started.pid -ErrorAction SilentlyContinue
            if ($process -and !$process.HasExited) {
                if ($process.Path -ne $started.executable -or $process.StartTime.ToUniversalTime() -ne ([DateTime]$started.start_time_utc).ToUniversalTime()) {
                    throw 'Started process identity changed; refusing to stop it.'
                }
                Stop-Process -Id $process.Id
                $process.WaitForExit()
            }
        }
        $row.finished_utc = [DateTime]::UtcNow.ToString('o')
        $results.Add($row)
        $results | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath "$out/results.json" -Encoding utf8
    }
    if ($Extended -and !$row.passed) { break }
}
if (@($results | Where-Object { !$_.passed }).Count) { exit 1 }
