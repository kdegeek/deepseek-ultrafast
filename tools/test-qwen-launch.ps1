# Offline checks by default. -LiveDefaults additionally reads local verified
# artifacts and health; neither mode loads a model or mutates a running process.
param([switch]$LiveDefaults)
$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'serve-qwen.ps1'
$tokens = $null
$parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($scriptPath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
foreach ($functionName in @('Assert-ManifestFiles','ConvertTo-QwenComparablePath','Test-QwenEngineProfile')) {
    $definition = $ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $functionName }, $true)
    . ([scriptblock]::Create($definition.Extent.Text))
}
$expectedProfile = @{
    executable = 'C:\workspace\target\release\qwen-resident.exe'
    library = 'C:\workspace\target\build-hip-mmq\bin\Release\qwen_native.dll'
    model_id = 'Qwen/Qwen3.8-27B'; quantization = 'Q8_0'; context = 65536; kv_type = 'q8_0'
    batch = 2048; ubatch = 512; threads = 8; flash_attention = $true; device_backend_pattern = 'ROCm*'
    mtp_draft = 0; dflash_draft = 3; draft_sampling = 'greedy'; draft_model = 'C:\models\café\draft.gguf'
    request_timeout_seconds = 10800; frozen_fallback = $false
}
$healthJson = @{
    state = 'ready'; request_limits = @{ deadline_seconds = 10800 }
    identity = @{
        model_id = 'Qwen/Qwen3.8-27B'; quantization = 'Q8_0'; context = 65536; kv_type = 'q8_0'
        batch = 2048; ubatch = 512; threads = 8; flash_attention = $true; device_backend = 'ROCm0'
        mmap = $false; lazy_weights = $false; mtp = $false; mtp_draft = 0; dflash_draft = 3; draft_sampling = 'greedy'
        draft_model_path = '\\?\C:\models\café\draft.gguf'
    }
} | ConvertTo-Json -Depth 5
$health = $healthJson | ConvertFrom-Json
$processPath = '\\?\C:\workspace\target\release\qwen-resident.exe'
$modulePaths = @('C:\Windows\System32\kernel32.dll', '\\?\C:\workspace\target\build-hip-mmq\bin\Release\qwen_native.dll')
if (!(Test-QwenEngineProfile $health $processPath $modulePaths $expectedProfile)) { throw 'Matching current profile with verbatim Unicode paths was rejected.' }
if (Test-QwenEngineProfile $health $processPath @('C:\workspace\target\build-hip\bin\Release\qwen_native.dll') $expectedProfile) { throw 'Wrong loaded backend variant matched.' }
if (Test-QwenEngineProfile $health $processPath @('C:\Windows\System32\kernel32.dll') $expectedProfile) { throw 'Absent native module matched.' }
if (Test-QwenEngineProfile $health 'C:\other\qwen-resident.exe' $modulePaths $expectedProfile) { throw 'Wrong executable matched.' }
foreach ($case in @(
    @{ scope = 'identity'; name = 'model_id'; value = 'another/model' },
    @{ scope = 'identity'; name = 'quantization'; value = 'Q4_K_M' },
    @{ scope = 'identity'; name = 'context'; value = 32768 },
    @{ scope = 'identity'; name = 'kv_type'; value = 'f16' },
    @{ scope = 'identity'; name = 'batch'; value = 1024 },
    @{ scope = 'identity'; name = 'ubatch'; value = 256 },
    @{ scope = 'identity'; name = 'threads'; value = 16 },
    @{ scope = 'identity'; name = 'flash_attention'; value = $false },
    @{ scope = 'identity'; name = 'device_backend'; value = 'Vulkan0' },
    @{ scope = 'identity'; name = 'mmap'; value = $true },
    @{ scope = 'identity'; name = 'lazy_weights'; value = $true },
    @{ scope = 'identity'; name = 'mtp_draft'; value = 1 },
    @{ scope = 'identity'; name = 'mtp_draft'; value = $null },
    @{ scope = 'identity'; name = 'dflash_draft'; value = 4 },
    @{ scope = 'identity'; name = 'dflash_draft'; value = $null },
    @{ scope = 'identity'; name = 'draft_sampling'; value = 'probabilistic' },
    @{ scope = 'identity'; name = 'draft_sampling'; value = $null },
    @{ scope = 'identity'; name = 'draft_model_path'; value = 'C:\models\different.gguf' },
    @{ scope = 'request_limits'; name = 'deadline_seconds'; value = 3600 },
    @{ scope = 'request_limits'; name = 'deadline_seconds'; value = $null }
)) {
    $changedHealth = $healthJson | ConvertFrom-Json
    $target = $changedHealth.($case.scope)
    $target.($case.name) = $case.value
    if (Test-QwenEngineProfile $changedHealth $processPath $modulePaths $expectedProfile) { throw "Mismatched $($case.name) matched the profile." }
}
$probabilisticHealth = $healthJson | ConvertFrom-Json
$probabilisticHealth.identity.draft_sampling = 'probabilistic'
$expectedProbabilistic = $expectedProfile.Clone()
$expectedProbabilistic.draft_sampling = 'probabilistic'
if (!(Test-QwenEngineProfile $probabilisticHealth $processPath $modulePaths $expectedProbabilistic)) { throw 'Matching probabilistic DFlash profile was rejected.' }
$missingFields = $healthJson | ConvertFrom-Json
$missingFields.request_limits = $null
$missingFields.identity.PSObject.Properties.Remove('mtp_draft')
$missingFields.identity.PSObject.Properties.Remove('dflash_draft')
$missingFields.identity.PSObject.Properties.Remove('draft_sampling')
if (Test-QwenEngineProfile $missingFields $processPath $modulePaths $expectedProfile) { throw 'Current engine with missing deadline/speculation fields matched.' }
$expectedFallback = $expectedProfile.Clone()
$expectedFallback.frozen_fallback = $true
$expectedFallback.executable = 'C:\workspace\target\fallback-ab990916\qwen-resident.exe'
$expectedFallback.library = 'C:\workspace\target\fallback-ab990916\qwen_native.dll'
$expectedFallback.quantization = 'Q4_K_M'
$expectedFallback.context = 32768
$expectedFallback.device_backend_pattern = 'Vulkan*'
$expectedFallback.mtp_draft = 0
$expectedFallback.dflash_draft = 0
$expectedFallback.request_timeout_seconds = 3600
$missingFields.identity.quantization = 'Q4_K_M'
$missingFields.identity.context = 32768
$missingFields.identity.device_backend = 'Vulkan0'
if (!(Test-QwenEngineProfile $missingFields $expectedFallback.executable @($expectedFallback.library) $expectedFallback)) { throw 'Frozen legacy health without new fields was rejected.' }
$expectedFallback.request_timeout_seconds = 7200
if (Test-QwenEngineProfile $missingFields $expectedFallback.executable @($expectedFallback.library) $expectedFallback) { throw 'Legacy fixed deadline was incorrectly treated as configurable.' }
$verbatimUnc = '\\?\UNC\server\share\qwen_native.dll'
if ((ConvertTo-QwenComparablePath $verbatimUnc) -ne $verbatimUnc) { throw 'Verbatim UNC namespace was incorrectly stripped.' }
$testDirectory = Join-Path ([IO.Path]::GetTempPath()) ("qwen-launch-test-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testDirectory | Out-Null
try {
    $inputFile = Join-Path $testDirectory 'compiled.cpp'
    [IO.File]::WriteAllText($inputFile, 'compiled source')
    $entry = [pscustomobject]@{ path = $inputFile; bytes = (Get-Item -LiteralPath $inputFile).Length; sha256 = (Get-FileHash -LiteralPath $inputFile -Algorithm SHA256).Hash.ToLowerInvariant() }
    $manifest = [pscustomobject]@{ files = @($entry) }
    Assert-ManifestFiles $manifest @($inputFile)
    $failed = $false
    try { Assert-ManifestFiles $manifest @((Join-Path $testDirectory 'uncovered.cpp')) }
    catch { $failed = $_.Exception.Message -like '*does not cover*' }
    if (!$failed) { throw 'Missing required source was not rejected.' }
    [IO.File]::WriteAllText($inputFile, 'modified source')
    $failed = $false
    try { Assert-ManifestFiles $manifest @($inputFile) }
    catch { $failed = $_.Exception.Message -like '*stale or modified*' }
    if (!$failed) { throw 'Modified build input was not rejected.' }
    $modelLock = [pscustomobject]@{ files = @([pscustomobject]@{ path = (Join-Path $testDirectory 'model-q8.lock.json'); bytes = 0; sha256 = 'not a compiled input' }) }
    Assert-ManifestFiles $modelLock @()
    foreach ($case in @(
        @{ args = @{ Port = 8081 }; expected = 'Port 8081 is reserved' },
        @{ args = @{ Profile = 'q4-fallback'; Quant = 'q8' }; expected = 'requires Q4' },
        @{ args = @{ Profile = 'q4-fallback'; Context = 65536 }; expected = 'outside the selected engine' },
        @{ args = @{ MtpDraft = 1; DFlashDraft = 3 }; expected = 'mutually exclusive' },
        @{ args = @{ DFlashDraft = 0; DraftSampling = 'probabilistic' }; expected = 'requires DFlash on the current engine' },
        @{ args = @{ MtpDraft = 1; DraftSampling = 'probabilistic' }; expected = 'requires DFlash on the current engine' },
        @{ args = @{ Profile = 'q4-fallback'; DFlashDraft = 3; DraftSampling = 'probabilistic' }; expected = 'requires DFlash on the current engine' },
        @{ args = @{ Batch = 32; UBatch = 64 }; expected = 'UBatch must not exceed' }
    )) {
        $failed = $false
        $caseArgs = $case.args
        try { & $scriptPath @caseArgs -Check | Out-Null }
        catch { $failed = $_.Exception.Message -like "*$($case.expected)*" }
        if (!$failed) { throw "Launcher did not reject invalid arguments: $($case.args -join ' ')" }
    }
    Write-Output 'PASS: launcher syntax, freshness, required inputs, model-lock separation, reserved port, fallback limits, invalid batch, loaded DLL variant, full health profile and legacy health compatibility.'
} finally {
    # Only explicitly created fixture files are removed; no recursive operation.
    Remove-Item -LiteralPath $inputFile -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $testDirectory
}
if ($LiveDefaults) {
    foreach ($case in @(
        @{ args=@{}; expected=@{quantization='Q8_0';dflash_draft=4;mtp_draft=0;draft_sampling='probabilistic';context=32768;request_timeout_seconds=3600} },
        @{ args=@{DFlashDraft=0}; expected=@{dflash_draft=0;mtp_draft=0;draft_sampling='greedy'} },
        @{ args=@{MtpDraft=0}; expected=@{dflash_draft=0;mtp_draft=0;draft_sampling='greedy'} },
        @{ args=@{MtpDraft=2}; expected=@{dflash_draft=0;mtp_draft=2;draft_sampling='greedy'} },
        @{ args=@{Quant='q4'}; expected=@{quantization='Q4_K_M';dflash_draft=0;draft_sampling='greedy'} },
        @{ args=@{Backend='HIP'}; expected=@{backend='HIP';dflash_draft=0;draft_sampling='greedy'} },
        @{ args=@{Profile='extended128'}; expected=@{context=131072;dflash_draft=4;request_timeout_seconds=7200} },
        @{ args=@{Profile='extended256'}; expected=@{context=262144;dflash_draft=4;request_timeout_seconds=10800} },
        @{ args=@{Context=70000;RequestTimeoutSeconds=5}; expected=@{context=70144;requested_context=70000;request_timeout_seconds=5} },
        @{ args=@{Profile='q4-fallback'}; expected=@{quantization='Q4_K_M';context=32768;dflash_draft=0;mtp_draft=0;draft_sampling='greedy';request_timeout_seconds=3600;frozen_fallback=$true} }
    )) {
        $caseArgs = $case.args
        $resolved = (& $scriptPath @caseArgs -Check | Out-String) | ConvertFrom-Json
        foreach ($key in $case.expected.Keys) {
            if ($resolved.$key -ne $case.expected[$key]) { throw "Resolved $key differs for $($case.args | ConvertTo-Json -Compress)." }
        }
    }
    Write-Output 'PASS: ten live read-only default/override/fallback profile combinations.'
}
