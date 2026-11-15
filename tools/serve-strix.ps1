param(
    [string]$SitePackages = "$env:USERPROFILE/.unsloth/studio/unsloth_studio/Lib/site-packages",
    [string]$Checkpoint = "$PSScriptRoot/../target/models/DeepSeek-V4.1-Flash",
    [ValidateRange(1, 65535)][int]$Port = 18081,
    [ValidateRange(3, 32)][int]$CacheGiB = 8,
    [ValidateRange(1, 128)][int]$Context = 128,
    [switch]$Build,
    [switch]$BuildOnly,
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath "$PSScriptRoot/..").Path
$SitePackages = (Resolve-Path -LiteralPath $SitePackages).Path
$Checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
$buildRoot = Join-Path $repoRoot 'target/strix-aten-server-build'
$shim = Join-Path $buildRoot 'strix_aten.dll'
$executable = Join-Path $repoRoot 'target/release/strix.exe'
$manifestPath = Join-Path $buildRoot 'serve-build.json'

function Get-SourceHashes {
    $inputs = @(
        (Join-Path $repoRoot 'Cargo.toml'),
        (Join-Path $repoRoot 'Cargo.lock'),
        (Join-Path $repoRoot 'crates/strix/Cargo.toml'),
        (Join-Path $repoRoot 'crates/strix/native/aten.cpp'),
        (Join-Path $repoRoot 'crates/strix/native/build-aten.ps1')
    )
    $inputs += Get-ChildItem -LiteralPath (Join-Path $repoRoot 'crates/strix/src') -Recurse -File | ForEach-Object { $_.FullName }
    $hashes = [ordered]@{}
    foreach ($inputPath in ($inputs | Sort-Object)) {
        $relative = [IO.Path]::GetRelativePath($repoRoot, $inputPath).Replace('\', '/')
        $hashes[$relative] = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
    }
    return $hashes
}

function Get-RuntimeHashes {
    $runtimeInputs = @(Get-ChildItem -LiteralPath (Join-Path $SitePackages 'torch/lib') -File -Filter '*.dll')
    foreach ($relative in @('_rocm_sdk_core/bin', '_rocm_sdk_libraries/bin')) {
        $runtimeInputs += Get-ChildItem -LiteralPath (Join-Path $SitePackages $relative) -Recurse -File
    }
    $hashes = [ordered]@{}
    foreach ($file in ($runtimeInputs | Sort-Object FullName)) {
        $key = [IO.Path]::GetRelativePath($SitePackages, $file.FullName).Replace('\', '/')
        $hashes[$key] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }
    return $hashes
}

function Test-HashMapsEqual($First, $Second) {
    if ($null -eq $First -or $null -eq $Second -or $First.Count -ne $Second.Count) { return $false }
    foreach ($key in $First.Keys) { if ($First[$key] -ne $Second[$key]) { return $false } }
    return $true
}

function Test-CurrentBuild {
    if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
        !(Test-Path -LiteralPath $executable -PathType Leaf) -or
        !(Test-Path -LiteralPath $shim -PathType Leaf)) { return $false }
    try {
        $saved = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
        if ($saved.site_packages -ne $SitePackages -or $saved.feature -ne 'native-serve') { return $false }
        $current = Get-SourceHashes
        if (!(Test-HashMapsEqual $saved.sources $current)) { return $false }
        if (!(Test-HashMapsEqual $saved.runtime (Get-RuntimeHashes))) { return $false }
        if ($saved.executable_sha256 -ne (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash -or
            $saved.shim_sha256 -ne (Get-FileHash -LiteralPath $shim -Algorithm SHA256).Hash) { return $false }
        # Also validate compiler/ABI inputs selected by the C++ build script.
        # Runtime hashes above include rocBLAS/hipBLAS and all their kernel data.
        $native = Get-Content -LiteralPath (Join-Path $buildRoot 'build-manifest.json') -Raw | ConvertFrom-Json
        if ($native.site_packages -ne $SitePackages) { return $false }
        foreach ($file in $native.files) {
            if (!(Test-Path -LiteralPath $file.path -PathType Leaf) -or
                $file.sha256 -ne (Get-FileHash -LiteralPath $file.path -Algorithm SHA256).Hash) { return $false }
        }
        return $true
    } catch { return $false }
}

$current = Test-CurrentBuild
if ($Check) {
    [ordered]@{ current = $current; executable = $executable; shim = $shim; site_packages = $SitePackages; checkpoint = $Checkpoint; port = $Port } | ConvertTo-Json
    if (!$current) { exit 1 }
    exit 0
}
if ($Build -or $BuildOnly -or !$current) {
    Write-Host 'Building the native ATen shim and Strix API from current sources.'
    $beforeSources = Get-SourceHashes
    $beforeRuntime = Get-RuntimeHashes
    Push-Location $repoRoot
    try {
        & pwsh -NoProfile -File (Join-Path $repoRoot 'crates/strix/native/build-aten.ps1') -SitePackages $SitePackages -Output $buildRoot
        if ($LASTEXITCODE -ne 0) { throw "ATen shim build failed: $LASTEXITCODE" }
        & cargo build -p deepseek-ultrafast-strix --release --features native-serve --bin strix
        if ($LASTEXITCODE -ne 0) { throw "Strix API build failed: $LASTEXITCODE" }
        $afterSources = Get-SourceHashes
        if (!(Test-HashMapsEqual $beforeSources $afterSources)) {
            throw 'Source files changed during compilation; rerun the launcher. No fresh build manifest was written.'
        }
        $afterRuntime = Get-RuntimeHashes
        if (!(Test-HashMapsEqual $beforeRuntime $afterRuntime)) {
            throw 'Native runtime files changed during compilation; rerun the launcher. No fresh build manifest was written.'
        }
        [ordered]@{
            built_utc = [DateTime]::UtcNow.ToString('o')
            feature = 'native-serve'
            site_packages = $SitePackages
            sources = $afterSources
            runtime = $afterRuntime
            executable_sha256 = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash
            shim_sha256 = (Get-FileHash -LiteralPath $shim -Algorithm SHA256).Hash
        } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    } finally { Pop-Location }
}
if ($BuildOnly) { return }

# Only this PowerShell process and its native child inherit the SDK selection.
$previousSite = $env:STRIX_ATEN_SITE_PACKAGES
$previousShim = $env:STRIX_ATEN_SHIM
try {
    $env:STRIX_ATEN_SITE_PACKAGES = $SitePackages
    $env:STRIX_ATEN_SHIM = $shim
    & $executable serve-aten $Checkpoint $Port $CacheGiB $Context
    if ($LASTEXITCODE -ne 0) { throw "Strix API exited: $LASTEXITCODE" }
} finally {
    $env:STRIX_ATEN_SITE_PACKAGES = $previousSite
    $env:STRIX_ATEN_SHIM = $previousShim
}
