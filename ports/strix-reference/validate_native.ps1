param(
    [string]$Checkpoint = 'target/models/DeepSeek-V4.1-Flash',
    [string]$Prompt = 'The capital of France is',
    [int]$MaxTokens = 3,
    [int]$CacheGiB = 8,
    [string]$OutputPrefix = 'target/strix-native-validation',
    [int[]]$ExpectedTokens = @(11111, 16, 1)
)
$ErrorActionPreference = 'Stop'
$binary = (Resolve-Path 'target/release/strix.exe').Path
$before = Get-CimInstance Win32_OperatingSystem
$info = [System.Diagnostics.ProcessStartInfo]::new()
$info.FileName = $binary
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
$info.RedirectStandardOutput = $true
$info.RedirectStandardError = $true
foreach ($argument in @('generate-experimental', $Checkpoint, $Prompt, "$MaxTokens", "$CacheGiB", '128')) {
    $info.ArgumentList.Add($argument)
}
$started = [DateTime]::UtcNow
$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $info
if (-not $process.Start()) { throw 'Native process did not start' }
$stdout = $process.StandardOutput.ReadToEndAsync()
$stderr = $process.StandardError.ReadToEndAsync()
$peakWorkingSet = 0L
$peakPrivate = 0L
$samples = 0
while (-not $process.WaitForExit(1000)) {
    $process.Refresh()
    $peakWorkingSet = [Math]::Max($peakWorkingSet, $process.PeakWorkingSet64)
    $peakPrivate = [Math]::Max($peakPrivate, $process.PrivateMemorySize64)
    $samples++
    if (($samples % 30) -eq 0) {
        Write-Output "Native validation running: $samples samples, peak working set $peakWorkingSet bytes"
    }
}
$output = $stdout.GetAwaiter().GetResult()
$errorOutput = $stderr.GetAwaiter().GetResult()
$output | Set-Content -LiteralPath "$OutputPrefix.json" -Encoding utf8
$errorOutput | Set-Content -LiteralPath "$OutputPrefix.log" -Encoding utf8
$after = Get-CimInstance Win32_OperatingSystem
$generation = if ($process.ExitCode -eq 0) { $output | ConvertFrom-Json -AsHashtable } else { $null }
$actualTokens = if ($null -ne $generation) { @($generation.generated_tokens) } else { @() }
$matches = ($actualTokens.Count -eq $ExpectedTokens.Count)
if ($matches) {
    for ($i = 0; $i -lt $ExpectedTokens.Count; $i++) {
        if ($actualTokens[$i] -ne $ExpectedTokens[$i]) { $matches = $false }
    }
}
$evidence = [ordered]@{
    checked_utc = [DateTime]::UtcNow.ToString('o')
    command = @($binary) + @($info.ArgumentList)
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash.ToLowerInvariant()
    native_exit_code = $process.ExitCode
    accepted = ($matches -and $process.ExitCode -eq 0)
    acceptance_rule = 'Exact original reference greedy token sequence, including EOS. No numerical tolerance or text-prefix substitution.'
    expected_tokens = $ExpectedTokens
    actual_tokens = $actualTokens
    elapsed_wall_seconds = ([DateTime]::UtcNow - $started).TotalSeconds
    process = @{
        pid = $process.Id
        peak_working_set_bytes = $peakWorkingSet
        peak_sampled_private_bytes = $peakPrivate
        samples = $samples
    }
    system = @{
        total_visible_physical_bytes = [long]$before.TotalVisibleMemorySize * 1024
        available_physical_before_bytes = [long]$before.FreePhysicalMemory * 1024
        available_physical_after_bytes = [long]$after.FreePhysicalMemory * 1024
        note = 'System-wide availability includes other processes and filesystem cache; it is not native model usage. Vulkan driver/shared GPU allocation is not isolated by process working set.'
    }
    generation = $generation
}
$evidence | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath "$OutputPrefix-evidence.json" -Encoding utf8
$process.Dispose()
Write-Output "Exact original-checkpoint acceptance: $($evidence.accepted); native exit: $($evidence.native_exit_code); tokens: $actualTokens"
if (-not $evidence.accepted) { exit 2 }
