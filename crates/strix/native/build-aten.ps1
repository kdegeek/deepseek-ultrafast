param(
    [Parameter(Mandatory=$true)][string]$SitePackages,
    [string]$Output = "$PSScriptRoot/../../../target/strix-aten-build",
    [string]$VcVars = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
)
$ErrorActionPreference = 'Stop'
$SitePackages = (Resolve-Path -LiteralPath $SitePackages).Path
$compiler = Join-Path $SitePackages '_rocm_sdk_core/lib/llvm/bin/clang-cl.exe'
$torch = Join-Path $SitePackages 'torch'
foreach ($path in @($compiler, $VcVars, "$torch/include/ATen/ATen.h", "$torch/lib/torch_cpu.lib", "$torch/lib/c10.lib")) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing native build input: $path" }
}
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Output = (Resolve-Path -LiteralPath $Output).Path
# A child cmd owns the VS environment. No parent/global PATH changes or installs.
$build = @"
@echo off
call "$VcVars"
if errorlevel 1 exit /b %errorlevel%
"$compiler" /nologo /std:c++17 /EHsc /MD /O2 /LD /D_CRT_SECURE_NO_WARNINGS /I"$torch/include" /I"$torch/include/torch/csrc/api/include" "$PSScriptRoot/aten.cpp" /Fo:"$Output/strix_aten.obj" /Fe:"$Output/strix_aten.dll" /link /LIBPATH:"$torch/lib" torch_cpu.lib c10.lib
exit /b %errorlevel%
"@
$script = Join-Path $Output 'build.cmd'
[IO.File]::WriteAllText($script, $build, [Text.Encoding]::ASCII)
Push-Location $Output
try { & cmd /d /c $script; if ($LASTEXITCODE -ne 0) { throw "Native shim build failed: $LASTEXITCODE" } }
finally { Pop-Location }
Get-FileHash -Algorithm SHA256 -LiteralPath "$Output/strix_aten.dll"
$inputs = @($compiler, "$torch/include/torch/csrc/api/include/torch/version.h",
    "$torch/lib/torch_cpu.dll", "$torch/lib/c10.dll", "$torch/lib/torch_hip.dll",
    "$SitePackages/_rocm_sdk_core/bin/amdhip64_7.dll", "$PSScriptRoot/aten.cpp", "$Output/strix_aten.dll")
$manifest = [ordered]@{ built_utc = [DateTime]::UtcNow.ToString('o'); site_packages = $SitePackages; files = @() }
foreach ($path in $inputs) {
    $file = Get-Item -LiteralPath $path
    $manifest.files += [ordered]@{ path = $file.FullName; bytes = $file.Length;
        file_version = $file.VersionInfo.FileVersion; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$Output/build-manifest.json" -Encoding utf8
