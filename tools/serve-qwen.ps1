param(
    [ValidateRange(1024,65535)][int]$Port = 18083,
    [ValidateRange(512,32768)][int]$Context = 32768,
    [ValidateRange(32,8192)][int]$Batch = 2048,
    [ValidateRange(32,8192)][int]$UBatch = 512,
    [ValidateRange(1,32)][int]$Threads = 8,
    [ValidateSet('q8_0','f16')][string]$KvType = 'q8_0',
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
$lockPath = Join-Path $root 'ports/qwen-resident/model.lock.json'
$lock = Get-Content -Raw -LiteralPath $lockPath | ConvertFrom-Json
$model = Join-Path $root "target/models/Qwen3.8-27B/$($lock.filename)"
$library = Join-Path $work 'build/bin/Release/qwen_native.dll'
$exe = Join-Path $root 'target/release/qwen-resident.exe'
$receiptPath = "$model.verified.json"
foreach ($path in @($model, $library, $exe, $receiptPath)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing $path. Run tools/build-qwen.ps1 -DownloadModel first." }
}
if ($Foreground -and $Background) { throw 'Choose only -Foreground or -Background.' }
if ($UBatch -gt $Batch) { throw 'UBatch must not exceed Batch.' }
if ($Batch -gt $Context) { throw 'Batch must not exceed Context.' }
if ($Port -eq 8081) { throw 'Port 8081 is reserved for Codex infrastructure.' }
$file = Get-Item -LiteralPath $model
$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
$mtime100ns = $file.LastWriteTimeUtc.Ticks - [DateTime]::UnixEpoch.Ticks
if ($file.Length -ne $lock.size_bytes -or $receipt.sha256 -ne $lock.sha256 -or
    $receipt.revision -ne $lock.revision -or $receipt.file_mtime_100ns -ne $mtime100ns -or
    $receipt.lock_sha256 -ne (Get-FileHash -Algorithm SHA256 -LiteralPath $lockPath).Hash.ToLowerInvariant()) {
    throw 'Model or lock changed since verification; run tools/download-qwen.py --verify-only.'
}
if ($VerifyModel -and (Get-FileHash -Algorithm SHA256 -LiteralPath $model).Hash.ToLowerInvariant() -ne $lock.sha256) { throw 'Model checksum mismatch.' }
$os = Get-CimInstance Win32_OperatingSystem
$availableGiB = [double]$os.FreePhysicalMemory / 1MB
# Conservative admission estimate, not residency evidence. Native allocation and
# populated-context acceptance are measured separately in docs/QWEN_STRIX.md.
$estimatedRuntimeGiB = [double]$lock.size_bytes / 1GB + 12
$summary = [ordered]@{
    model_id = $lock.model_id; model = $model; model_sha256 = $receipt.sha256
    library = $library; executable = $exe; context = $Context; kv_type = $KvType
    port = $Port; available_physical_gib = [Math]::Round($availableGiB, 2)
    estimated_runtime_gib = [Math]::Round($estimatedRuntimeGiB, 2)
    minimum_remaining_physical_gib = $MinimumFreeGiB
}
if ($Check) {
    try { $summary.health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 }
    catch { $summary.health = 'not running' }
    $summary | ConvertTo-Json -Depth 8
    return
}
if ($availableGiB - $estimatedRuntimeGiB -lt $MinimumFreeGiB) {
    throw "Insufficient RAM headroom: $([Math]::Round($availableGiB,2)) GiB available; need $([Math]::Round($estimatedRuntimeGiB + $MinimumFreeGiB,2)) GiB before start. Close other memory-intensive work or retry later."
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { throw "Port $Port is already in use; no process was stopped." }
$arguments = @('--model', $model, '--library', $library, '--port', "$Port", '--context', "$Context", '--batch', "$Batch", '--ubatch', "$UBatch", '--threads', "$Threads", '--kv-type', $KvType, '--gpu-layers', '999')
if ($NoFlashAttention) { $arguments += '--no-flash-attention' }
if ($Foreground) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Qwen exited: $LASTEXITCODE" }
    return
}
New-Item -ItemType Directory -Force -Path "$work/runs" | Out-Null
$run = Join-Path $work "runs/$([DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff'))"
New-Item -ItemType Directory -Path $run | Out-Null
$quoted = $arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
$process = Start-Process -FilePath $exe -ArgumentList $quoted -WindowStyle Hidden -PassThru -WorkingDirectory $root -RedirectStandardOutput "$run/stdout.log" -RedirectStandardError "$run/stderr.log"
$identity = [ordered]@{ pid = $process.Id; start_time_utc = $process.StartTime.ToUniversalTime().ToString('o'); executable = $exe; port = $Port; run_directory = $run }
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
