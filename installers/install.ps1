# RandBLAS autoinstaller for native Windows (MSVC).
#
# Builds RandBLAS and the dependencies it needs into a self-contained
# "RandNLA-project" directory, the same layout install.sh produces on Linux and
# macOS:
#   lib:     dependency sources
#   install: RandBLAS-install and the dependency installs
#   build:   one build directory per project above
#
# Nothing is installed system-wide, no PATH entry is created, and your
# environment is not modified unless you pass -ModifyEnvironment.
#
# You bring Visual Studio (or the Build Tools), CMake and Git, in an x64
# developer shell. This script does not install a toolchain; when one is
# missing or wrong it says so and tells you how to fix it.
#
# Prerequisites and supported configurations are in INSTALL.md.

[CmdletBinding()]
param(
    # Where dependencies, builds and installs go. Defaults to
    # $env:RANDNLA_PROJECT_DIR when set -- which is what lets this installer
    # and RandLAPACK's share one dependency tree -- and otherwise to a
    # RandNLA-project directory beside this clone.
    [string] $ProjectDir = "",

    # Where the dependency stack lives. Defaults to <ProjectDir>\install. CI
    # points this at a cache shared with the core workflow.
    [string] $DependencyRoot = "",

    # Install RandBLAS itself here instead of <ProjectDir>\install\RandBLAS-install.
    # Dependencies still go in the project directory.
    [string] $Prefix = "",

    [int]    $Jobs = 0,
    [switch] $Fresh,
    [switch] $SkipTests,
    [switch] $Examples,
    [switch] $ModifyEnvironment,
    [switch] $Yes
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir   = Split-Path -Parent $scriptDir

#==============================================================================
# Toolchain preflight.
#
# The architecture guard is the important one. "Developer PowerShell for VS"
# and "Developer Command Prompt for VS" both default to an *x86* toolchain, and
# an x86 linker cannot use the x64 import libraries every BLAS backend here
# ships. Without this check the failure surfaces three layers down as BLAS++
# reporting "BLAS library not found", which blames the libraries when the
# compiler is at fault.
#==============================================================================
. (Join-Path $repoDir ".github\scripts\windows\toolchain-arch.ps1")

$missing = @()
foreach ($tool in @("cl.exe", "cmake.exe", "git.exe")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { $missing += $tool }
}
if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "PREREQUISITE MISSING: $($missing -join ', ') not found on PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "  RandBLAS needs Visual Studio (or the Build Tools) with the C++ workload,"
    Write-Host "  plus CMake and Git, in an x64 developer shell."
    Write-Host ""
    Write-Host "  Open 'x64 Native Tools Command Prompt for VS 2022' from the Start menu,"
    Write-Host "  or run this in any Command Prompt to configure one:"
    Write-Host ""
    Write-Host '    for /f "usebackq delims=" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvars64.bat"'
    Write-Host ""
    Write-Host "  See INSTALL.md for the full prerequisite list."
    exit 1
}

$arch = Get-ClTargetArchitecture
$archProblem = Get-ToolchainArchitectureProblem $arch
if ($archProblem) {
    Write-Host ""
    Write-Host "WRONG TOOLCHAIN ARCHITECTURE" -ForegroundColor Red
    Write-Host ""
    Write-Host "  $archProblem"
    Write-Host ""
    exit 1
}

#==============================================================================
# Interactivity.
#
# Prompts happen only when someone is there to answer: not with -Yes, and not
# when stdin is redirected. Every question has a defensible unattended default
# so an automated run cannot hang.
#==============================================================================
$script:Interactive = -not $Yes -and -not [Console]::IsInputRedirected `
    -and [Environment]::UserInteractive

function Read-YesNo {
    param([string] $Question, [bool] $Default)
    if (-not $script:Interactive) { return $Default }
    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    while ($true) {
        $reply = (Read-Host "$Question $suffix").Trim()
        if ($reply -eq "") { return $Default }
        if ($reply -match '^(y|yes)$') { return $true }
        if ($reply -match '^(n|no)$')  { return $false }
    }
}

if ($Jobs -le 0) {
    $Jobs = [Environment]::ProcessorCount
}

#==============================================================================
# Project layout.
#
# Precedence matches install.sh exactly: the flag, then RANDNLA_PROJECT_DIR,
# then a sibling of this clone. Honouring the environment variable is what
# lets a machine that has already run RandLAPACK's installer reuse its BLAS++
# rather than building a second copy.
#==============================================================================
if (-not $ProjectDir) {
    if ($env:RANDNLA_PROJECT_DIR) {
        $ProjectDir = $env:RANDNLA_PROJECT_DIR
    } else {
        $ProjectDir = Join-Path (Split-Path $repoDir -Parent) "RandNLA-project"
    }
}
$ProjectDir = [IO.Path]::GetFullPath($ProjectDir)

# Deep dependency build trees plus MSVC's own path limits make long project
# paths fail in ways that are hard to attribute, so warn before the build
# rather than after.
if ($ProjectDir.Length -gt 150) {
    Write-Warning ("ProjectDir is $($ProjectDir.Length) characters long. Deep dependency " +
        "build trees may exceed Windows path limits; consider something shorter, such as C:\RandNLA.")
}

if (-not $DependencyRoot) { $DependencyRoot = Join-Path $ProjectDir "install" }
$DependencyRoot = [IO.Path]::GetFullPath($DependencyRoot)

$installDir = if ($Prefix) {
    [IO.Path]::GetFullPath($Prefix)
} else {
    Join-Path $ProjectDir "install\RandBLAS-install"
}
$buildDir = Join-Path $ProjectDir "build\RandBLAS-build"

foreach ($d in @($ProjectDir, $DependencyRoot, (Join-Path $ProjectDir "lib"), (Join-Path $ProjectDir "build"))) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
if ($Fresh -and (Test-Path -LiteralPath $buildDir)) {
    Remove-Item -Recurse -Force -LiteralPath $buildDir
}
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

Write-Host ""
Write-Host "RandBLAS installer" -ForegroundColor Cyan
Write-Host "  toolchain     x64 ($arch)"
Write-Host "  project dir   $ProjectDir"
Write-Host "  dependencies  $DependencyRoot"
Write-Host "  install to    $installDir"
Write-Host ""

#==============================================================================
# Dependencies.
#
# Delegated to the same provisioner CI uses, so there is one implementation of
# "fetch oneMKL, build BLAS++, build GoogleTest" rather than two that drift.
# It pins every source to an immutable ref and records provenance, so a
# dependency is reused only when it came from what we would fetch now.
#==============================================================================
$setup = Join-Path $repoDir ".github\actions\setup-randblas-deps-windows\setup.ps1"
$setupArgs = @{ DependencyRoot = $DependencyRoot }
if ($Examples) { $setupArgs["InstallLapackpp"] = $true }

Write-Host "[1/4] Provisioning dependencies (oneMKL, BLAS++, Random123, GoogleTest) ..."
# No $LASTEXITCODE check: setup.ps1 is a PowerShell script that sets
# $ErrorActionPreference = "Stop" and throws, so a failure propagates on its
# own. Reading $LASTEXITCODE here would be worse than redundant -- it is unset
# until some native command runs, and Set-StrictMode turns reading an unset
# variable into an error. That made the whole installer fail on exactly the
# runs where every dependency was already cached and no native command had run.
& $setup @setupArgs

#==============================================================================
# RandBLAS.
#==============================================================================
$cmakeArgs = @(
    "-S", $repoDir,
    "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$($installDir.Replace('\','/'))",
    "-Dblaspp_DIR=$($env:blaspp_DIR)",
    "-DRandom123_DIR=$($env:Random123_DIR)"
)
if ($SkipTests) {
    $cmakeArgs += "-DBUILD_TESTS=OFF"
} else {
    $cmakeArgs += @("-DBUILD_TESTS=ON", "-DGTest_ROOT=$($env:googletest_PREFIX)")
}

# Ninja is not guaranteed present outside a full Visual Studio install; fall
# back to the NMake generator the CI provisioner already uses.
if (-not (Get-Command "ninja.exe" -ErrorAction SilentlyContinue)) {
    $cmakeArgs[5] = "NMake Makefiles"
}

Write-Host "[2/4] Configuring RandBLAS ..."
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }

Write-Host "[3/4] Building and installing RandBLAS ..."
& cmake --build $buildDir -j $Jobs --target install
if ($LASTEXITCODE -ne 0) { throw "Build failed." }

#==============================================================================
# Verification.
#
# Compile, link and run a program against the finished install. Configuring is
# not the same as producing something that works: this catches a BLAS that
# resolves at configure time but fails to link, and a runtime DLL that was
# never staged beside the executable.
#==============================================================================
Write-Host "[4/4] Verifying the install links and runs ..."
$conftest = Join-Path $ProjectDir "build\conftest"
if (Test-Path -LiteralPath $conftest) { Remove-Item -Recurse -Force -LiteralPath $conftest }
New-Item -ItemType Directory -Force -Path (Join-Path $conftest "src") | Out-Null

Set-Content -Path (Join-Path $conftest "src\CMakeLists.txt") -Encoding ascii -Value @(
    'cmake_minimum_required(VERSION 3.21)',
    'project(randblas_conftest CXX)',
    'find_package(RandBLAS REQUIRED)',
    'add_executable(conftest conftest.cc)',
    'target_link_libraries(conftest RandBLAS)',
    'randblas_stage_runtime_dlls(conftest)')

Set-Content -Path (Join-Path $conftest "src\conftest.cc") -Encoding ascii -Value @(
    '#include <RandBLAS.hh>',
    '#include <blas.hh>',
    '#include <blas/defines.h>',
    '#include <cstdio>',
    '#include <cmath>',
    '#include <vector>',
    'int main() {',
    '#if defined(BLAS_ILP64)',
    '    std::printf("blas_ilp64=1\n");',
    '#else',
    '    std::printf("blas_ilp64=0\n");',
    '#endif',
    '    const int64_t m = 8, n = 4;',
    '    std::vector<double> S(m * n);',
    '    RandBLAS::DenseDist D(m, n);',
    '    RandBLAS::RNGState state(0);',
    '    RandBLAS::fill_dense(D, S.data(), state);',
    '    std::vector<double> C(n * n, 0.0);',
    '    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,',
    '               n, n, m, 1.0, S.data(), m, S.data(), m, 0.0, C.data(), n);',
    '    for (int64_t i = 0; i < n; ++i) {',
    '        if (!(C[i + i * n] > 0.0) || !std::isfinite(C[i + i * n])) return 1;',
    '    }',
    '    std::printf("OK\n");',
    '    return 0;',
    '}')

$conftestGenerator = if (Get-Command "ninja.exe" -ErrorAction SilentlyContinue) { "Ninja" } else { "NMake Makefiles" }
& cmake -S (Join-Path $conftest "src") -B (Join-Path $conftest "build") -G $conftestGenerator `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_PREFIX_PATH=$($installDir.Replace('\','/'))" `
    "-Dblaspp_DIR=$($env:blaspp_DIR)" `
    "-DRandom123_DIR=$($env:Random123_DIR)" | Out-Host
if ($LASTEXITCODE -ne 0) { throw "The verification program failed to configure." }
& cmake --build (Join-Path $conftest "build") | Out-Host
if ($LASTEXITCODE -ne 0) { throw "The verification program failed to build." }

$conftestExe = Get-ChildItem -LiteralPath (Join-Path $conftest "build") -Recurse -Filter "conftest.exe" |
    Select-Object -First 1
if (-not $conftestExe) { throw "The verification program built but produced no executable." }
$conftestOutput = & $conftestExe.FullName
if ($LASTEXITCODE -ne 0 -or ($conftestOutput -notcontains "OK")) {
    throw "The verification program ran but did not produce a correct result:`n$($conftestOutput -join "`n")"
}
$observedWidth = if ($conftestOutput -contains "blas_ilp64=1") {
    "ILP64 (64-bit BLAS integers)"
} else {
    "LP64 (32-bit BLAS integers)"
}

#==============================================================================
# Optional: persist RANDNLA_PROJECT_DIR.
#
# Opt-in, mirroring install.sh's --modify-rc. SetEnvironmentVariable at User
# scope is the Windows equivalent of appending to a shell profile, and the only
# mechanism that survives a new shell.
#==============================================================================
if ($ModifyEnvironment) {
    [Environment]::SetEnvironmentVariable("RANDNLA_PROJECT_DIR", $ProjectDir, "User")
    Write-Host ""
    Write-Host "Set RANDNLA_PROJECT_DIR=$ProjectDir for your user account (open a new shell to pick it up)."
}

#==============================================================================
# Summary.
#==============================================================================
Write-Host ""
Write-Host "RandBLAS installed successfully." -ForegroundColor Green
Write-Host ""
Write-Host "  Backend            oneMKL, $observedWidth"
Write-Host "  Project layout     $ProjectDir"
Write-Host "  Installed library  $installDir"
Write-Host ""
if (-not $SkipTests) {
    Write-Host "  Run the test suite:"
    Write-Host "    ctest --test-dir $buildDir"
    Write-Host ""
}
Write-Host "  Consume from CMake with:"
Write-Host "    -DRandBLAS_DIR=$($installDir.Replace('\','/'))/lib/cmake/RandBLAS"
if (-not $ModifyEnvironment) {
    Write-Host ""
    Write-Host "  To have other RandNLA installers reuse these dependencies, set:"
    Write-Host "    setx RANDNLA_PROJECT_DIR `"$ProjectDir`""
    Write-Host "  (or re-run with -ModifyEnvironment)"
}

if (-not $Examples) {
    Write-Host ""
    Write-Host "  The examples are not built by default: they additionally need"
    Write-Host "  LAPACK++ and fast_matrix_market, and they require OpenMP."
    $buildNow = Read-YesNo "  Build them now?" $false
    if (-not $buildNow) {
        Write-Host "  To build them later, re-run with -Examples:"
        Write-Host "    powershell -ExecutionPolicy Bypass -File $scriptDir\install.ps1 -Examples -ProjectDir `"$ProjectDir`""
        Write-Host ""
        exit 0
    }
    $Examples = $true
    & $setup -DependencyRoot $DependencyRoot -InstallLapackpp
}

if ($Examples) {
    $examplesBuild = Join-Path $ProjectDir "build\examples-build"
    Write-Host ""
    Write-Host "Configuring and building examples ..."
    & cmake -S (Join-Path $repoDir "examples") -B $examplesBuild -G $conftestGenerator `
        "-DCMAKE_BUILD_TYPE=Release" `
        "-DCMAKE_PREFIX_PATH=$($installDir.Replace('\','/'))" `
        "-Dblaspp_DIR=$($env:blaspp_DIR)" `
        "-Dlapackpp_DIR=$($env:lapackpp_DIR)" `
        "-DRandom123_DIR=$($env:Random123_DIR)" `
        "-DFETCHCONTENT_BASE_DIR=$($ProjectDir.Replace('\','/'))/build/fetchcontent-cache" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Examples failed to configure." }
    & cmake --build $examplesBuild -j $Jobs | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Examples failed to build." }
    Write-Host ""
    Write-Host "Examples built: $examplesBuild" -ForegroundColor Green
}

Write-Host ""
