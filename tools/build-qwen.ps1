param(
    [string]$CMake,
    [string]$Ninja,
    [string]$VcVars = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat',
    [ValidateRange(1,64)][int]$Jobs = 12,
    [switch]$BootstrapOnly,
    [switch]$SkipRust,
    [switch]$DownloadModel
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath "$PSScriptRoot/..").Path
$work = Join-Path $root 'target/qwen-resident'
$lock = Get-Content -Raw -LiteralPath "$root/ports/qwen-resident/backend.lock.json" | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path "$work/downloads" | Out-Null

function Invoke-Checked([scriptblock]$Action) {
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

$source = Join-Path $work 'llama.cpp'
if (!(Test-Path -LiteralPath "$source/.git")) {
    if ((Test-Path -LiteralPath $source) -and (Get-ChildItem -LiteralPath $source -Force)) { throw "Refusing to replace existing non-git source: $source" }
    Invoke-Checked { git init $source }
    Invoke-Checked { git -C $source remote add origin $lock.repository }
    Invoke-Checked { git -C $source fetch --depth 1 origin $lock.revision }
    Invoke-Checked { git -C $source checkout --detach FETCH_HEAD }
}
$head = (& git -C $source rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $lock.revision) { throw "Backend source is not pinned revision $($lock.revision): $head" }
if (& git -C $source status --porcelain --untracked-files=no) { throw 'Backend source has tracked edits; preserve them and use a clean pinned source.' }

$sdk = Join-Path $work "vulkan-sdk/$($lock.vulkan_sdk.version)"
if (!(Test-Path -LiteralPath "$sdk/Bin/glslc.exe")) {
    $installer = Join-Path $work "downloads/vulkansdk-windows-X64-$($lock.vulkan_sdk.version).exe"
    if (!(Test-Path -LiteralPath $installer)) {
        Invoke-Checked { curl.exe --fail --location --retry 3 --output $installer $lock.vulkan_sdk.url }
    }
    if ((Get-Item -LiteralPath $installer).Length -ne $lock.vulkan_sdk.size_bytes -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash.ToLowerInvariant() -ne $lock.vulkan_sdk.sha256) { throw 'Vulkan SDK installer checksum mismatch.' }
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'LunarG') { throw 'Vulkan SDK publisher signature is invalid.' }
    # Official copy-only mode deliberately avoids registry, layers, shortcuts and PATH changes.
    $copy = Start-Process -FilePath $installer -WindowStyle Hidden -PassThru -Wait -ArgumentList @(
        '--root', "`"$sdk`"", '--accept-licenses', '--default-answer', '--confirm-command', 'install', 'copy_only=1'
    ) -RedirectStandardOutput "$work/downloads/sdk-copy.stdout.log" -RedirectStandardError "$work/downloads/sdk-copy.stderr.log"
    if ($copy.ExitCode -ne 0) { throw "Vulkan SDK copy-only failed: $($copy.ExitCode)" }
}
foreach ($path in @("$sdk/Bin/glslc.exe", "$sdk/Include/vulkan/vulkan.h", "$sdk/Lib/vulkan-1.lib", "$sdk/Lib/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake")) {
    if (!(Test-Path -LiteralPath $path)) { throw "Incomplete isolated Vulkan SDK: $path" }
}

$strataTools = Join-Path $env:USERPROFILE 'Documents/Codex/2026-10-24/ca/work/Strata/.rocm-win'
if (!$CMake) {
    $candidate = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($candidate) { $CMake = $candidate.Source }
    elseif (Test-Path -LiteralPath "$strataTools/Lib/site-packages/cmake/data/bin/cmake.exe") { $CMake = "$strataTools/Lib/site-packages/cmake/data/bin/cmake.exe" }
}
if (!$Ninja) {
    $candidate = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($candidate) { $Ninja = $candidate.Source }
    elseif (Test-Path -LiteralPath "$strataTools/Scripts/ninja.exe") { $Ninja = "$strataTools/Scripts/ninja.exe" }
}
foreach ($path in @($CMake, $Ninja, $VcVars)) {
    if (!$path -or !(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing build tool: '$path'; supply -CMake, -Ninja or -VcVars." }
}
if ($DownloadModel) {
    $python = Join-Path $env:USERPROFILE '.unsloth/studio/unsloth_studio/Scripts/python.exe'
    if (!(Test-Path -LiteralPath $python)) { throw 'Run tools/download-qwen.py with an installed Python 3 interpreter.' }
    Invoke-Checked { & $python "$root/tools/download-qwen.py" }
}
if ($BootstrapOnly) {
    [ordered]@{ source = $source; revision = $head; vulkan_sdk = $sdk; cmake = $CMake; ninja = $Ninja } | ConvertTo-Json
    return
}
$build = Join-Path $work 'build'
New-Item -ItemType Directory -Force -Path $build | Out-Null
$ninjaDirectory = Split-Path -Parent $Ninja
$cmakeDirectory = Split-Path -Parent $CMake
$ctest = Join-Path $cmakeDirectory 'ctest.exe'
# The child shell owns the VS environment. No global or parent PATH modification.
$script = @"
@echo off
call "$VcVars"
if errorlevel 1 exit /b %errorlevel%
set "PATH=$ninjaDirectory;$cmakeDirectory;%PATH%"
set "VULKAN_SDK=$sdk"
"$CMake" -S "$root/ports/qwen-resident" -B "$build" -G "Ninja Multi-Config" -DCMAKE_MAKE_PROGRAM="$Ninja" -DLLAMA_SOURCE_DIR="$source" -DGGML_VULKAN=ON -DCMAKE_PREFIX_PATH="$sdk" -DVulkan_INCLUDE_DIR="$sdk/Include" -DVulkan_LIBRARY="$sdk/Lib/vulkan-1.lib" -DVulkan_GLSLC_EXECUTABLE="$sdk/Bin/glslc.exe"
if errorlevel 1 exit /b %errorlevel%
"$CMake" --build "$build" --config Release --target qwen_native qwen_stream_test -j $Jobs
if errorlevel 1 exit /b %errorlevel%
"$ctest" --test-dir "$build" -C Release --output-on-failure
exit /b %errorlevel%
"@
$command = Join-Path $work 'build-native.cmd'
[IO.File]::WriteAllText($command, $script, [Text.Encoding]::ASCII)
Invoke-Checked { & cmd /d /c $command }
if (!$SkipRust) {
    Push-Location $root
    try { Invoke-Checked { cargo build --release -p deepseek-ultrafast-qwen --features native-qwen --bin qwen-resident } }
    finally { Pop-Location }
}
$manifest = [ordered]@{ built_utc = [DateTime]::UtcNow.ToString('o'); backend_revision = $head; backend = 'Vulkan'; cmake_version = (& $CMake --version | Select-Object -First 1); rust_built = !$SkipRust; files = @() }
$inputs = @($CMake, $Ninja, "$sdk/Bin/glslc.exe", "$sdk/Lib/vulkan-1.lib")
$inputs += @(Get-ChildItem -LiteralPath "$root/ports/qwen-resident" -File | ForEach-Object FullName)
$inputs += @(Get-ChildItem -LiteralPath "$build/bin/Release" -Filter '*.dll' | ForEach-Object FullName)
if (!$SkipRust) {
    $inputs += @("$root/Cargo.toml", "$root/Cargo.lock", "$root/crates/qwen-resident/Cargo.toml", "$root/target/release/qwen-resident.exe")
    $inputs += @(Get-ChildItem -LiteralPath "$root/crates/qwen-resident/src" -File -Recurse | ForEach-Object FullName)
}
foreach ($path in $inputs) {
    $file = Get-Item -LiteralPath $path
    $manifest.files += [ordered]@{ path = $file.FullName; bytes = $file.Length; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant() }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$work/build-manifest.json" -Encoding utf8
Write-Host "Built Qwen native backend: $build/bin/Release/qwen_native.dll"
