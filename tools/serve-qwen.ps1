param(
    [ValidateRange(1024,65535)][int]$Port = 18083,
    [ValidateSet('fast','extended64','extended128','extended256','q4-fallback')][string]$Profile = 'fast',
    [ValidateSet('q4','q8')][string]$Quant = 'q8',
    [ValidateSet('Vulkan','HIP')][string]$Backend = 'Vulkan',
    [ValidatePattern('^[a-z0-9-]+$')][string]$BuildVariant,
    [ValidateRange(0,262144)][int]$Context = 0,
    [ValidateRange(32,8192)][int]$Batch = 2048,
    [ValidateRange(32,8192)][int]$UBatch = 512,
    [ValidateRange(1,32)][int]$Threads = 8,
    [ValidateSet('q8_0','f16')][string]$KvType = 'q8_0',
    [ValidateRange(0,3)][int]$MtpDraft = 0,
    [ValidateRange(0,7)][int]$DFlashDraft = 0,
    [ValidateSet('greedy','probabilistic')][string]$DraftSampling = 'greedy',
    [ValidateRange(1,21600)][int]$RequestTimeoutSeconds = 3600,
    [switch]$NoFlashAttention,
    [switch]$Foreground,
    [switch]$Background,
    [switch]$Check,
    [switch]$VerifyModel,
    [ValidateRange(20,80)][int]$MinimumFreeGiB = 20,
    [ValidateRange(30,1800)][int]$ReadyTimeoutSeconds = 300
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath "$PSScriptRoot/..").Path
$work = Join-Path $root 'target/qwen-resident'
$fallback = $Profile -eq 'q4-fallback'
# The measured interactive winner is Vulkan Q8 with DFlash4. An explicit
# speculative selector, including zero, overrides this profile default.
if (!$fallback -and $Quant -eq 'q8' -and $Backend -eq 'Vulkan' -and
    !$PSBoundParameters.ContainsKey('MtpDraft') -and !$PSBoundParameters.ContainsKey('DFlashDraft')) {
    $DFlashDraft = 4
}
if (!$PSBoundParameters.ContainsKey('DraftSampling') -and !$fallback -and
    $Quant -eq 'q8' -and $Backend -eq 'Vulkan' -and $DFlashDraft -gt 0) {
    $DraftSampling = 'probabilistic'
}
$DraftSampling = $DraftSampling.ToLowerInvariant()
if ($DraftSampling -eq 'probabilistic' -and ($DFlashDraft -eq 0 -or $fallback)) {
    throw 'Probabilistic draft sampling requires DFlash on the current engine.'
}
$deviceBackendPattern = if ($Backend -eq 'HIP') { 'ROCm*' } else { 'Vulkan*' }
if ($fallback) {
    if (($PSBoundParameters.ContainsKey('Quant') -and $Quant -ne 'q4') -or $Backend -ne 'Vulkan' -or $MtpDraft -ne 0 -or $DFlashDraft -ne 0 -or $BuildVariant) {
        throw 'The preserved q4-fallback profile requires Q4, Vulkan and speculative decoding disabled.'
    }
    if ($RequestTimeoutSeconds -ne 3600) { throw 'The preserved fallback has a fixed one-hour request deadline.' }
    $Quant = 'q4'
}
if ($Context -eq 0) {
    $Context = @{ fast = 32768; extended64 = 65536; extended128 = 131072; extended256 = 262144; 'q4-fallback' = 32768 }[$Profile]
}
if ($Context -lt 512 -or ($fallback -and $Context -gt 32768)) { throw 'Context is outside the selected engine profile capacity.' }
$requestedContext = $Context
# The pinned backend pads a single sequence to 256 tokens. Resolve that before
# allocation estimates and identity readback so custom sizes are represented
# honestly instead of killing a healthy rounded context after loading weights.
$Context = [int]([Math]::Ceiling($Context / 256.0) * 256)
if (!$PSBoundParameters.ContainsKey('RequestTimeoutSeconds') -and !$fallback) {
    $RequestTimeoutSeconds = if ($Context -gt 131072) { 10800 } elseif ($Context -gt 65536) { 7200 } else { 3600 }
}
if ($Foreground -and $Background) { throw 'Choose only -Foreground or -Background.' }
if ($MtpDraft -gt 0 -and $DFlashDraft -gt 0) { throw 'MTP and DFlash drafting are mutually exclusive.' }
if ($UBatch -gt $Batch) { throw 'UBatch must not exceed Batch.' }
if ($Batch -gt $Context) { throw 'Batch must not exceed Context.' }
if ($Port -eq 8081) { throw 'Port 8081 is reserved for Codex infrastructure.' }
$lockName = if ($Quant -eq 'q8') { 'model-q8.lock.json' } else { 'model.lock.json' }
$lockPath = Join-Path $root "ports/qwen-resident/$lockName"
$lock = Get-Content -Raw -LiteralPath $lockPath | ConvertFrom-Json
$model = Join-Path $root "target/models/Qwen3.8-27B/$($lock.filename)"
$buildDirectory = if ($Backend -eq 'HIP') { 'build-hip' } else { 'build' }
if ($BuildVariant) { $buildDirectory += "-$BuildVariant" }
$manifestName = "$buildDirectory-manifest.json"
$library = Join-Path $work "$buildDirectory/bin/Release/qwen_native.dll"
$exe = Join-Path $root 'target/release/qwen-resident.exe'
if ($fallback) {
    $library = Join-Path $work 'fallback-ab990916/qwen_native.dll'
    $exe = Join-Path $work 'fallback-ab990916/qwen-resident.exe'
}
$receiptPath = "$model.verified.json"
foreach ($path in @($model, $library, $exe, $receiptPath)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing $path. Build the selected backend and run tools/download-qwen.py --quant $Quant first." }
}
$file = Get-Item -LiteralPath $model
$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
$mtime100ns = $file.LastWriteTimeUtc.Ticks - [DateTime]::UnixEpoch.Ticks
if ($file.Length -ne $lock.size_bytes -or $receipt.sha256 -ne $lock.sha256 -or
    $receipt.revision -ne $lock.revision -or $receipt.repository -ne $lock.repository -or
    $receipt.filename -ne $lock.filename -or $receipt.size_bytes -ne $lock.size_bytes -or
    $receipt.file_mtime_100ns -ne $mtime100ns -or
    $receipt.lock_sha256 -ne (Get-FileHash -Algorithm SHA256 -LiteralPath $lockPath).Hash.ToLowerInvariant()) {
    throw "Model or lock changed since verification; run tools/download-qwen.py --quant $Quant --verify-only."
}
if ($VerifyModel -and (Get-FileHash -Algorithm SHA256 -LiteralPath $model).Hash.ToLowerInvariant() -ne $lock.sha256) { throw 'Model checksum mismatch.' }
$draftModel = $null
$draftLock = $null
if ($DFlashDraft -gt 0) {
    $draftLockPath = Join-Path $root 'ports/qwen-resident/model-dflash2.lock.json'
    $draftLock = Get-Content -Raw -LiteralPath $draftLockPath | ConvertFrom-Json
    if ($draftLock.target_model_id -ne $lock.model_id) { throw 'Pinned DFlash drafter does not match the selected target model.' }
    $draftModel = Join-Path $root "target/models/Qwen3.8-27B/$($draftLock.filename)"
    $draftReceipt = Get-Content -Raw -LiteralPath "$draftModel.verified.json" | ConvertFrom-Json
    $draftFile = Get-Item -LiteralPath $draftModel
    $draftMtime100ns = $draftFile.LastWriteTimeUtc.Ticks - [DateTime]::UnixEpoch.Ticks
    if ($draftFile.Length -ne $draftLock.size_bytes -or $draftReceipt.sha256 -ne $draftLock.sha256 -or
        $draftReceipt.revision -ne $draftLock.revision -or $draftReceipt.repository -ne $draftLock.repository -or
        $draftReceipt.filename -ne $draftLock.filename -or $draftReceipt.size_bytes -ne $draftLock.size_bytes -or
        $draftReceipt.file_mtime_100ns -ne $draftMtime100ns -or
        $draftReceipt.lock_sha256 -ne (Get-FileHash -Algorithm SHA256 -LiteralPath $draftLockPath).Hash.ToLowerInvariant()) {
        throw 'DFlash model or lock changed since verification; run tools/download-qwen.py --lock ports/qwen-resident/model-dflash2.lock.json --verify-only.'
    }
    if ($VerifyModel -and (Get-FileHash -Algorithm SHA256 -LiteralPath $draftModel).Hash.ToLowerInvariant() -ne $draftLock.sha256) { throw 'DFlash model checksum mismatch.' }
}

function Assert-ManifestFiles($Manifest, [string[]]$Required) {
    $covered = @{}
    foreach ($entry in $Manifest.files) {
        # Model receipts bind their own locks; quant selection is not a compiled
        # input and must not invalidate an otherwise matching native build.
        if ([IO.Path]::GetFileName($entry.path) -in 'model.lock.json','model-q8.lock.json','model-dflash2.lock.json') { continue }
        if (!(Test-Path -LiteralPath $entry.path -PathType Leaf)) { throw "Build input or output is missing: $($entry.path)" }
        $item = Get-Item -LiteralPath $entry.path
        if ($item.Length -ne $entry.bytes -or (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.path).Hash.ToLowerInvariant() -ne $entry.sha256) {
            throw "Build is stale or modified: $($entry.path). Rebuild the selected backend."
        }
        $covered[$item.FullName] = $true
    }
    foreach ($path in $Required) {
        $full = [IO.Path]::GetFullPath($path)
        if (!$covered.ContainsKey($full)) { throw "Build manifest does not cover required input/output: $full" }
    }
}

function ConvertTo-QwenComparablePath([string]$Path) {
    if ([string]::IsNullOrEmpty($Path)) { return $null }
    # Windows reports some loaded modules in the verbatim drive namespace.
    # Normalize only drive paths; preserve UNC semantics and Unicode.
    if ($Path.StartsWith('\\?\') -and $Path.Length -gt 6 -and $Path[5] -eq ':') {
        $Path = $Path.Substring(4)
    }
    return [IO.Path]::GetFullPath($Path)
}

function Test-QwenEngineProfile($Health, [string]$ProcessPath, [string[]]$ModulePaths, $Expected) {
    if ($Health.state -ne 'ready' -or !$Health.identity) { return $false }
    if ((ConvertTo-QwenComparablePath $ProcessPath) -ne (ConvertTo-QwenComparablePath $Expected.executable)) { return $false }
    $loadedNative = @($ModulePaths | Where-Object { [IO.Path]::GetFileName($_) -ieq 'qwen_native.dll' } | ForEach-Object { ConvertTo-QwenComparablePath $_ })
    if ($loadedNative.Count -ne 1 -or $loadedNative[0] -ne (ConvertTo-QwenComparablePath $Expected.library)) { return $false }
    $actual = $Health.identity
    foreach ($name in @('model_id','quantization','context','kv_type','batch','ubatch','threads','flash_attention')) {
        if ($null -eq $actual.$name -or $actual.$name -ne $Expected.$name) { return $false }
    }
    if ($actual.device_backend -notlike $Expected.device_backend_pattern -or
        $actual.mmap -ne $false -or $actual.lazy_weights -ne $false) { return $false }
    $actualMtp = $actual.mtp_draft
    $actualDFlash = $actual.dflash_draft
    $actualDraftSampling = $actual.draft_sampling
    $actualDeadline = $Health.request_limits.deadline_seconds
    if ($Expected.frozen_fallback) {
        # The hash-pinned old binary predates these health fields and has a
        # fixed one-hour deadline with speculative decoding disabled.
        if ($null -eq $actualMtp) { $actualMtp = 0 }
        if ($null -eq $actualDFlash) { $actualDFlash = 0 }
        if ($null -eq $actualDraftSampling) { $actualDraftSampling = 'greedy' }
        if ($null -eq $actualDeadline) { $actualDeadline = 3600 }
        if ($actual.mtp -eq $true) { return $false }
    }
    if ($null -eq $actualMtp -or $null -eq $actualDFlash -or $null -eq $actualDraftSampling -or $null -eq $actualDeadline -or
        $actualMtp -ne $Expected.mtp_draft -or $actualDFlash -ne $Expected.dflash_draft -or
        $actualDraftSampling -ne $Expected.draft_sampling -or
        $actualDeadline -ne $Expected.request_timeout_seconds) { return $false }
    if ($Expected.dflash_draft -gt 0 -and
        (ConvertTo-QwenComparablePath $actual.draft_model_path) -ne (ConvertTo-QwenComparablePath $Expected.draft_model)) { return $false }
    return $true
}

if ($fallback) {
    # Frozen binaries are intentionally independent of the current source tree.
    $expected = @{
        'qwen-resident.exe' = 'd80510c60bd70a6a40a06f7710ef6ac28d9566b2456829b68b4ff132757c9f35'
        'qwen_native.dll' = '80617b5f6c7d9b877f372e67088ef99c8266269cd127ed199e9cca2f806f948e'
        'build-manifest.json' = 'ca31a785ae7e7d85136e3742f8f6fc247ecfe34dcd919245bdbcbec035b1edc1'
    }
    foreach ($name in $expected.Keys) {
        $path = Join-Path $work "fallback-ab990916/$name"
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant() -ne $expected[$name]) {
            throw "Preserved fallback checksum mismatch: $name"
        }
    }
} else {
    $manifestPath = Join-Path $work $manifestName
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $backendLock = Get-Content -Raw -LiteralPath "$root/ports/qwen-resident/backend.lock.json" | ConvertFrom-Json
    if ($manifest.backend -ne $Backend -or $manifest.backend_revision -ne $backendLock.revision) { throw 'Build manifest backend or source pin does not match.' }
    $nativeInputs = @(Get-ChildItem -LiteralPath "$root/ports/qwen-resident" -File | Where-Object { $_.Extension -in '.cpp','.h','.hpp' -or $_.Name -in 'CMakeLists.txt','backend.lock.json' } | ForEach-Object FullName)
    $rustInputs = @("$root/Cargo.toml", "$root/Cargo.lock", "$root/crates/qwen-resident/Cargo.toml", $exe)
    $rustInputs += @(Get-ChildItem -LiteralPath "$root/crates/qwen-resident/src" -Recurse -File | ForEach-Object FullName)
    $required = $nativeInputs + @($library) + @($manifest.runtime.preload_libraries | Where-Object { $_ })
    if ($manifest.rust_built) {
        Assert-ManifestFiles $manifest ($required + $rustInputs)
    } else {
        Assert-ManifestFiles $manifest $required
        # SkipRust must retain a genuine earlier Cargo build record; hashing fresh
        # sources beside an old executable does not establish build freshness.
        $rustManifestPath = Join-Path $work 'rust-build-manifest.json'
        if (!(Test-Path -LiteralPath $rustManifestPath)) { $rustManifestPath = Join-Path $work 'build-manifest.json' }
        $rustManifest = Get-Content -Raw -LiteralPath $rustManifestPath | ConvertFrom-Json
        if (!$rustManifest.rust_built) { throw 'No genuine Rust build record; rebuild without -SkipRust.' }
        $rustOnly = [pscustomobject]@{ files = @($rustManifest.files | Where-Object { [IO.Path]::GetFullPath($_.path) -in @($rustInputs | ForEach-Object { [IO.Path]::GetFullPath($_) }) }) }
        Assert-ManifestFiles $rustOnly $rustInputs
    }
}
$os = Get-CimInstance Win32_OperatingSystem
$availableGiB = [double]$os.FreePhysicalMemory / 1MB
# Conservative admission estimate, not residency evidence. Native allocation and
# populated-context acceptance are measured separately in docs/QWEN_STRIX.md.
$kvBytesPerElement = if ($KvType -eq 'f16') { 2.0 } else { 34.0 / 32.0 }
# 16 full-attention layers x 4 KV heads x 256 dimensions x K and V.
$estimatedKvGiB = [double]$Context * 16 * 4 * 256 * 2 * $kvBytesPerElement / 1GB
# Packed weights plus KV, compute/host/recurrent allowance and MTP scratch.
$estimatedRuntimeGiB = [double]$lock.size_bytes / 1GB + $estimatedKvGiB + 10 + $(if ($MtpDraft -gt 0) { 2 } else { 0 })
if ($DFlashDraft -gt 0) { $estimatedRuntimeGiB += [double]$draftLock.size_bytes / 1GB + $estimatedKvGiB + 2 }
$summary = [ordered]@{
    model_id = $lock.model_id; model = $model; model_sha256 = $receipt.sha256
    library = $library; executable = $exe; context = $Context; requested_context = $requestedContext; kv_type = $KvType
    profile = $Profile; quantization = $lock.quantization; backend = $Backend; build_variant = $BuildVariant; mtp_draft = $MtpDraft
    dflash_draft = $DFlashDraft; draft_sampling = $DraftSampling; draft_model = $draftModel; draft_model_sha256 = $draftLock.sha256
    request_timeout_seconds = $RequestTimeoutSeconds; frozen_fallback = $fallback
    estimated_attention_kv_gib = [Math]::Round($estimatedKvGiB, 2)
    port = $Port; available_physical_gib = [Math]::Round($availableGiB, 2)
    estimated_runtime_gib = [Math]::Round($estimatedRuntimeGiB, 2)
    minimum_remaining_physical_gib = $MinimumFreeGiB
}
$expectedProfile = @{
    executable = $exe; library = $library; model_id = $lock.model_id; quantization = $lock.quantization
    context = $Context; kv_type = $KvType; batch = $Batch; ubatch = $UBatch; threads = $Threads
    flash_attention = !$NoFlashAttention; device_backend_pattern = $deviceBackendPattern
    mtp_draft = $MtpDraft; dflash_draft = $DFlashDraft; draft_sampling = $DraftSampling; draft_model = $draftModel
    request_timeout_seconds = $RequestTimeoutSeconds; frozen_fallback = $fallback
}
if ($Check) {
    $summary.health_matches_selected_profile = $false
    try {
        $summary.health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
        $running = Get-Process -Id ([int]$summary.health.pid) -ErrorAction Stop
        $summary.health_matches_selected_profile = Test-QwenEngineProfile $summary.health $running.Path @($running.Modules | ForEach-Object FileName) $expectedProfile
    }
    catch { if (!$summary.Contains('health')) { $summary.health = 'not running' } }
    $summary | ConvertTo-Json -Depth 8
    return
}
$residentProcesses = @(Get-Process -Name 'qwen-resident' -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -and $_.Path.StartsWith((Join-Path $root 'target') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
})
if ($residentProcesses.Count) {
    throw "An engine from this workspace is already running (PID $($residentProcesses.Id -join ', ')); no second full model was loaded and no process was stopped."
}
if ($availableGiB - $estimatedRuntimeGiB -lt $MinimumFreeGiB) {
    throw "Insufficient RAM headroom: $([Math]::Round($availableGiB,2)) GiB available; need $([Math]::Round($estimatedRuntimeGiB + $MinimumFreeGiB,2)) GiB before start. Close other memory-intensive work or retry later."
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { throw "Port $Port is already in use; no process was stopped." }
$arguments = @('--model', $model, '--library', $library, '--port', "$Port", '--context', "$Context", '--batch', "$Batch", '--ubatch', "$UBatch", '--threads', "$Threads", '--kv-type', $KvType, '--gpu-layers', '999')
if (!$fallback) { $arguments += @('--mtp-draft', "$MtpDraft", '--draft-sampling', $DraftSampling, '--request-timeout-seconds', "$RequestTimeoutSeconds") }
if ($DFlashDraft -gt 0) { $arguments += @('--draft-model', $draftModel, '--dflash-draft', "$DFlashDraft") }
if (!$fallback -and $Backend -eq 'HIP') {
    foreach ($directory in $manifest.runtime.dll_directories) {
        if (!(Test-Path -LiteralPath $directory -PathType Container)) { throw "Missing private HIP DLL directory: $directory" }
        $arguments += @('--dll-directory', $directory)
    }
    foreach ($preload in $manifest.runtime.preload_libraries) {
        $arguments += @('--preload-library', $preload)
    }
    if (!$manifest.runtime.dll_directories -or !$manifest.runtime.preload_libraries) { throw 'HIP manifest has no private runtime directory/preload configuration.' }
}
if ($NoFlashAttention) { $arguments += '--no-flash-attention' }
if ($Foreground) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Qwen exited: $LASTEXITCODE" }
    return
}
New-Item -ItemType Directory -Force -Path "$work/runs" | Out-Null
$run = Join-Path $work "runs/$([DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff'))"
New-Item -ItemType Directory -Path $run | Out-Null
# Preserve the exact verified build record before a later experiment rebuilds
# the same directory. A path plus backend commit alone is not binary identity.
$runManifest = if ($fallback) { Join-Path $work 'fallback-ab990916/build-manifest.json' } else { $manifestPath }
Copy-Item -LiteralPath $runManifest -Destination "$run/build-manifest.json"
if (!$fallback -and !$manifest.rust_built) { Copy-Item -LiteralPath $rustManifestPath -Destination "$run/rust-build-manifest.json" }
Copy-Item -LiteralPath $receiptPath -Destination "$run/model-receipt.json"
if ($draftModel) { Copy-Item -LiteralPath "$draftModel.verified.json" -Destination "$run/draft-model-receipt.json" }
$quoted = $arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
$process = Start-Process -FilePath $exe -ArgumentList $quoted -WindowStyle Hidden -PassThru -WorkingDirectory $root -RedirectStandardOutput "$run/stdout.log" -RedirectStandardError "$run/stderr.log"
$identity = [ordered]@{ pid = $process.Id; start_time_utc = $process.StartTime.ToUniversalTime().ToString('o'); executable = $exe; port = $Port; run_directory = $run }
$identity.library = $library
$identity.executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
$identity.library_sha256 = (Get-FileHash -LiteralPath $library -Algorithm SHA256).Hash.ToLowerInvariant()
$identity.build_manifest_sha256 = (Get-FileHash -LiteralPath "$run/build-manifest.json" -Algorithm SHA256).Hash.ToLowerInvariant()
$identity.model_sha256 = $receipt.sha256
$identity | ConvertTo-Json | Set-Content -LiteralPath "$run/process.json" -Encoding utf8
$deadline = [DateTime]::UtcNow.AddSeconds($ReadyTimeoutSeconds)
try {
    while ([DateTime]::UtcNow -lt $deadline) {
        $process.Refresh()
        if ($process.HasExited) { throw "Qwen exited with $($process.ExitCode); see $run/stderr.log" }
        $freeGiB = [double](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
        if ($freeGiB -lt $MinimumFreeGiB) { throw "RAM headroom dropped below $MinimumFreeGiB GiB during startup." }
        $health = $null
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        } catch { }
        if ($null -ne $health) {
            if ($health.pid -ne $process.Id) { throw 'Readiness responded from an unexpected process.' }
            if ($health.state -eq 'failed') { throw "Native initialization failed: $($health.error); see $run/stderr.log" }
            if ($health.state -eq 'ready') {
                $ready = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/readyz" -TimeoutSec 2
                if (!(Test-QwenEngineProfile $health $process.Path @($process.Modules | ForEach-Object FileName) $expectedProfile)) {
                    throw 'Ready engine identity does not match requested resident model/profile.'
                }
                $identity.health = $health
                $identity.base_url = "http://127.0.0.1:$Port/v1"
                $identity | ConvertTo-Json -Depth 8
                return
            }
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Qwen readiness timed out; see $run/stderr.log"
} catch {
    # Only the exact process created above can be stopped on failed startup.
    $process.Refresh()
    if (!$process.HasExited) { $process.Kill(); $process.WaitForExit() }
    throw
}
