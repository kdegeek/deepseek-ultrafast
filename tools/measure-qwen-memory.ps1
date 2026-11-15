<#
.SYNOPSIS
Read-only bounded Windows physical/commit, process I/O and GPU memory sampling.
.DESCRIPTION
Attach to the Qwen API PID, or pass a PID explicitly. Never launches or stops a
process. Raw page-I/O counters are system-wide; process I/O includes all file and
device traffic and is not a model-file-specific trace. GPU allocations may share
physical RAM with process/system accounting on this UMA machine: do not sum them.
Use a new output path for each load/idle/decode observation window.
#>
[CmdletBinding()]
param(
    [int]$ServerPid = 0,
    [ValidateRange(1, 65535)][int]$Port = 18083,
    [ValidateRange(1, 3600)][int]$DurationSeconds = 120,
    [ValidateRange(0.25, 30)][double]$IntervalSeconds = 1,
    [ValidateRange(1, 100)][double]$MinimumAvailableGiB = 20,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$StopFile = ''
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw "Output already exists; preserve previous evidence: $Output" }
$outputPath = [IO.Path]::GetFullPath($Output)
$parentPath = [IO.Path]::GetDirectoryName($outputPath)
[IO.Directory]::CreateDirectory($parentPath) | Out-Null
$initialHealth = $null
if ($ServerPid -eq 0) {
    $initialHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
    $ServerPid = [int]$initialHealth.pid
    if ($ServerPid -le 0) { throw 'The API did not return a valid PID.' }
}
$observedProcess = Get-Process -Id $ServerPid
$observedStart = $observedProcess.StartTime.ToUniversalTime()
$observedExecutable = $observedProcess.Path
if (!$observedExecutable -or [IO.Path]::GetFileNameWithoutExtension($observedExecutable) -ne 'qwen-resident') {
    throw "Expected qwen-resident.exe, found '$observedExecutable'. No process was changed."
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace QwenObservation {
    [StructLayout(LayoutKind.Sequential)]
    public struct MemoryStatus {
        public uint Length;
        public uint Load;
        public ulong TotalPhysical;
        public ulong AvailablePhysical;
        public ulong TotalPageFile;
        public ulong AvailablePageFile;
        public ulong TotalVirtual;
        public ulong AvailableVirtual;
        public ulong AvailableExtendedVirtual;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct ProcessMemory {
        public uint Size;
        public uint PageFaultCount;
        public UIntPtr PeakWorkingSetSize;
        public UIntPtr WorkingSetSize;
        public UIntPtr QuotaPeakPagedPoolUsage;
        public UIntPtr QuotaPagedPoolUsage;
        public UIntPtr QuotaPeakNonPagedPoolUsage;
        public UIntPtr QuotaNonPagedPoolUsage;
        public UIntPtr PagefileUsage;
        public UIntPtr PeakPagefileUsage;
        public UIntPtr PrivateUsage;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct IoCounters {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }
    public static class Native {
        [DllImport("kernel32.dll", SetLastError=true)]
        public static extern bool GlobalMemoryStatusEx(ref MemoryStatus value);
        [DllImport("psapi.dll", SetLastError=true)]
        public static extern bool GetProcessMemoryInfo(IntPtr process, ref ProcessMemory value, uint size);
        [DllImport("kernel32.dll", SetLastError=true)]
        public static extern bool GetProcessIoCounters(IntPtr process, out IoCounters value);
    }
}
'@

$counterPaths = @(
    '\Memory\Committed Bytes', '\Memory\Commit Limit',
    '\Memory\Page Reads/sec', '\Memory\Page Writes/sec',
    '\Memory\Pages Input/sec', '\Memory\Pages Output/sec'
)
$gpuPaths = @(
    '\GPU Process Memory(*)\Dedicated Usage',
    '\GPU Process Memory(*)\Shared Usage',
    '\GPU Process Memory(*)\Total Committed'
)
$gpuAvailable = $true
$gpuError = $null
try { $null = Get-Counter -Counter $gpuPaths -ErrorAction Stop }
catch { $gpuAvailable = $false; $gpuError = $_.Exception.Message }
$report = [ordered]@{
    schema_version = 1
    started_utc = [DateTime]::UtcNow.ToString('o')
    process_id = $ServerPid
    process_started_utc = $observedStart.ToString('o')
    executable = $observedExecutable
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $observedExecutable).Hash.ToLowerInvariant()
    initial_health = $initialHealth
    requested_duration_seconds = $DurationSeconds
    interval_seconds = $IntervalSeconds
    minimum_available_gib = $MinimumAvailableGiB
    gpu_counters_available = $gpuAvailable
    gpu_counter_error = $gpuError
    interpretation = @(
        'System page I/O counters include unrelated processes; page faults include soft faults.'
        'Process I/O counters include all process file/device I/O, not just the model file.'
        'GPU committed/dedicated/shared usage may overlap physical/process accounting on UMA; never add these views together.'
        'Allocation counts, capacity headroom and observed I/O are evidence for this window, not a guarantee against future Windows paging.'
        'No process, pagefile, driver, service, or OS configuration is changed by this observer.'
    )
    samples = [Collections.Generic.List[object]]::new()
    pressure_observed = $false
    stop_reason = $null
}
function Save-Observation {
    $temporary = "$outputPath.tmp"
    [IO.File]::WriteAllText($temporary, ($report | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    [IO.File]::Move($temporary, $outputPath, $true)
}
function Get-RawCounterValue($samples, [string]$suffix) {
    $entry = $samples | Where-Object { $_.Path.EndsWith($suffix, [StringComparison]::OrdinalIgnoreCase) } | Select-Object -First 1
    if ($null -eq $entry) { return $null }
    return [uint64]$entry.RawValue
}
function Get-GpuCounterValue($samples, [string]$suffix) {
    $entries = @($samples | Where-Object { $_.Path -match "(?i)gpu process memory\(pid_${ServerPid}_" -and $_.Path.EndsWith($suffix, [StringComparison]::OrdinalIgnoreCase) })
    if (!$entries.Count) { return $null }
    return [uint64](($entries | Measure-Object -Property CookedValue -Sum).Sum)
}
function Get-MaximumGiB($values) {
    $present = @($values | Where-Object { $null -ne $_ })
    if (!$present.Count) { return $null }
    return [Math]::Round(($present | Measure-Object -Maximum).Maximum / 1GB, 3)
}
$timer = [Diagnostics.Stopwatch]::StartNew()
try {
    while ($timer.Elapsed.TotalSeconds -lt $DurationSeconds) {
        if ($StopFile -and (Test-Path -LiteralPath $StopFile)) { $report.stop_reason = 'stop_file'; break }
        $sampleStarted = $timer.Elapsed.TotalSeconds
        $observedProcess.Refresh()
        if ($observedProcess.HasExited) { $report.stop_reason = 'observed_process_exited'; break }
        if ($observedProcess.StartTime.ToUniversalTime() -ne $observedStart) { throw 'PID was reused; observation stopped.' }
        $systemMemory = [QwenObservation.MemoryStatus]::new()
        $systemMemory.Length = [Runtime.InteropServices.Marshal]::SizeOf($systemMemory)
        if (![QwenObservation.Native]::GlobalMemoryStatusEx([ref]$systemMemory)) { throw 'GlobalMemoryStatusEx failed.' }
        $processMemory = [QwenObservation.ProcessMemory]::new()
        $processMemory.Size = [Runtime.InteropServices.Marshal]::SizeOf($processMemory)
        if (![QwenObservation.Native]::GetProcessMemoryInfo($observedProcess.Handle, [ref]$processMemory, $processMemory.Size)) { throw 'GetProcessMemoryInfo failed.' }
        $processIo = [QwenObservation.IoCounters]::new()
        if (![QwenObservation.Native]::GetProcessIoCounters($observedProcess.Handle, [ref]$processIo)) { throw 'GetProcessIoCounters failed.' }
        $counters = @((Get-Counter -Counter $counterPaths -ErrorAction Stop).CounterSamples)
        if ($gpuAvailable) {
            try { $counters += (Get-Counter -Counter $gpuPaths -ErrorAction Stop).CounterSamples }
            catch {
                # Optional PDH GPU providers can lose instances while unrelated
                # applications exit. Keep physical/process observations usable.
                $report.gpu_counter_error = $_.Exception.Message
            }
        }
        $sample = [ordered]@{
            utc = [DateTime]::UtcNow.ToString('o')
            elapsed_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
            system = [ordered]@{
                physical_total_bytes = $systemMemory.TotalPhysical
                physical_available_bytes = $systemMemory.AvailablePhysical
                memory_load_percent = $systemMemory.Load
                committed_bytes = Get-RawCounterValue $counters '\memory\committed bytes'
                commit_limit_bytes = Get-RawCounterValue $counters '\memory\commit limit'
                page_read_operations_raw = Get-RawCounterValue $counters '\memory\page reads/sec'
                page_write_operations_raw = Get-RawCounterValue $counters '\memory\page writes/sec'
                pages_input_raw = Get-RawCounterValue $counters '\memory\pages input/sec'
                pages_output_raw = Get-RawCounterValue $counters '\memory\pages output/sec'
            }
            process = [ordered]@{
                working_set_bytes = $processMemory.WorkingSetSize.ToUInt64()
                peak_working_set_bytes = $processMemory.PeakWorkingSetSize.ToUInt64()
                private_committed_bytes = $processMemory.PrivateUsage.ToUInt64()
                pagefile_usage_bytes = $processMemory.PagefileUsage.ToUInt64()
                page_faults_raw = [uint64]$processMemory.PageFaultCount
                read_bytes_raw = $processIo.ReadTransferCount
                write_bytes_raw = $processIo.WriteTransferCount
                other_bytes_raw = $processIo.OtherTransferCount
                read_operations_raw = $processIo.ReadOperationCount
                write_operations_raw = $processIo.WriteOperationCount
            }
            gpu = [ordered]@{
                dedicated_usage_bytes = Get-GpuCounterValue $counters '\dedicated usage'
                shared_usage_bytes = Get-GpuCounterValue $counters '\shared usage'
                total_committed_bytes = Get-GpuCounterValue $counters '\total committed'
            }
            below_memory_reserve = ($systemMemory.AvailablePhysical -lt $MinimumAvailableGiB * 1GB)
        }
        if ($sample.below_memory_reserve) { $report.pressure_observed = $true }
        $report.samples.Add($sample)
        Save-Observation
        if ($sample.below_memory_reserve -or $report.samples.Count -eq 1 -or $report.samples.Count % 30 -eq 0) {
            [ordered]@{ elapsed_seconds = $sample.elapsed_seconds; available_gib = [Math]::Round($systemMemory.AvailablePhysical / 1GB, 3); process_working_set_gib = [Math]::Round($processMemory.WorkingSetSize.ToUInt64() / 1GB, 3); gpu_committed_gib = if ($null -ne $sample.gpu.total_committed_bytes) { [Math]::Round($sample.gpu.total_committed_bytes / 1GB, 3) } else { $null }; below_memory_reserve = $sample.below_memory_reserve } | ConvertTo-Json -Compress | Write-Output
        }
        $remaining = $IntervalSeconds - ($timer.Elapsed.TotalSeconds - $sampleStarted)
        if ($remaining -gt 0) { Start-Sleep -Milliseconds ([int]($remaining * 1000)) }
    }
    if (!$report.stop_reason) { $report.stop_reason = 'bounded_duration_elapsed' }
}
catch {
    $report.stop_reason = 'observer_error'
    $report.observer_error = $_.Exception.Message
}
finally {
    $report.finished_utc = [DateTime]::UtcNow.ToString('o')
    $report.elapsed_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
    if ($report.samples.Count -gt 0) {
        $first = $report.samples[0]
        $last = $report.samples[$report.samples.Count - 1]
        $summary = [ordered]@{
            samples = $report.samples.Count
            minimum_available_gib = [Math]::Round(($report.samples | ForEach-Object { $_.system.physical_available_bytes } | Measure-Object -Minimum).Minimum / 1GB, 3)
            maximum_system_commit_gib = [Math]::Round(($report.samples | ForEach-Object { $_.system.committed_bytes } | Measure-Object -Maximum).Maximum / 1GB, 3)
            maximum_process_working_set_gib = [Math]::Round(($report.samples | ForEach-Object { $_.process.working_set_bytes } | Measure-Object -Maximum).Maximum / 1GB, 3)
            maximum_process_private_commit_gib = [Math]::Round(($report.samples | ForEach-Object { $_.process.private_committed_bytes } | Measure-Object -Maximum).Maximum / 1GB, 3)
            maximum_gpu_dedicated_gib = Get-MaximumGiB @($report.samples | ForEach-Object { $_.gpu.dedicated_usage_bytes })
            maximum_gpu_shared_gib = Get-MaximumGiB @($report.samples | ForEach-Object { $_.gpu.shared_usage_bytes })
            maximum_gpu_committed_gib = Get-MaximumGiB @($report.samples | ForEach-Object { $_.gpu.total_committed_bytes })
            window_process_read_bytes = [decimal]$last.process.read_bytes_raw - [decimal]$first.process.read_bytes_raw
            window_process_write_bytes = [decimal]$last.process.write_bytes_raw - [decimal]$first.process.write_bytes_raw
            window_process_page_faults = [decimal]$last.process.page_faults_raw - [decimal]$first.process.page_faults_raw
            window_system_page_read_operations = [decimal]$last.system.page_read_operations_raw - [decimal]$first.system.page_read_operations_raw
            window_system_page_write_operations = [decimal]$last.system.page_write_operations_raw - [decimal]$first.system.page_write_operations_raw
            window_system_pages_input = [decimal]$last.system.pages_input_raw - [decimal]$first.system.pages_input_raw
            window_system_pages_output = [decimal]$last.system.pages_output_raw - [decimal]$first.system.pages_output_raw
            pressure_observed = $report.pressure_observed
        }
        $report.summary = $summary
    }
    Save-Observation
    [ordered]@{ output = $outputPath; stop_reason = $report.stop_reason; summary = $report.summary; observer_error = $report.observer_error } | ConvertTo-Json -Depth 5 | Write-Output
}
if ($report.stop_reason -eq 'observer_error') { exit 1 }
if ($report.pressure_observed) { exit 2 }
