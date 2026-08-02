[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("CoreOpenMP", "CoreSerial", "Downstream", "Examples")]
    [string] $Task,

    [string] $SourceRoot = "",
    [string] $WorkRoot = "",
    [string] $DependencyRoot = "",
    [string] $VcpkgExecutable = "",
    [switch] $SetupDependencies,
    [switch] $SanitizeAddress
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string] $Program,
        [Parameter(Mandatory = $false)][string[]] $Arguments = @()
    )

    Write-Host "+ $Program $($Arguments -join ' ')"
    & $Program @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE."
    }
}

function Convert-ToCMakePath {
    param([Parameter(Mandatory = $true)][string] $Path)
    return $Path.Replace("\", "/")
}

function Require-EnvironmentVariable {
    param([Parameter(Mandatory = $true)][string] $Name)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if (-not $value) {
        throw "Required environment variable $Name is not set."
    }
    return $value
}

if (-not $SourceRoot) {
    if ($env:GITHUB_WORKSPACE) {
        $SourceRoot = $env:GITHUB_WORKSPACE
    }
    else {
        $SourceRoot = [IO.Path]::GetFullPath(
            (Join-Path $PSScriptRoot "..\..\..")
        )
    }
}
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)

if (-not $WorkRoot) {
    $WorkRoot = Join-Path (Split-Path $SourceRoot -Parent) "RandBLAS-windows-ci"
}
$WorkRoot = [IO.Path]::GetFullPath($WorkRoot)
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null

if ($SetupDependencies) {
    if (-not $DependencyRoot) {
        $DependencyRoot = Join-Path $WorkRoot "dependencies"
    }
    $setupScript = Join-Path $SourceRoot `
        ".github\actions\setup-randblas-deps-windows\setup.ps1"
    $setupArguments = @{
        DependencyRoot = $DependencyRoot
        InstallLapackpp = ($Task -eq "Examples")
        SanitizeAddress = $SanitizeAddress
    }
    if ($VcpkgExecutable) {
        $setupArguments["VcpkgExecutable"] = $VcpkgExecutable
    }
    & $setupScript @setupArguments
}

$blasppDir = Require-EnvironmentVariable "blaspp_DIR"
$mklRoot = Require-EnvironmentVariable "MKLROOT"
$mklBin = Join-Path $mklRoot "bin"
if (-not (Test-Path -LiteralPath $mklBin)) {
    throw "oneMKL runtime directory does not exist: $mklBin"
}
$env:PATH = "$mklBin;$env:PATH"

function Install-RandBLAS {
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [Parameter(Mandatory = $true)][bool] $BuildTests,
        [Parameter(Mandatory = $true)][bool] $UseOpenMP,
        [bool] $UseNativeDependencyPaths = $false,
        [bool] $SanitizeAddress = $false
    )

    $build = Join-Path $WorkRoot "$Name-build"
    $install = Join-Path $WorkRoot "$Name-install"
    $configuredBlasppDir = $blasppDir
    if ($UseNativeDependencyPaths) {
        $configuredBlasppDir = $blasppDir.Replace("/", "\")
    }
    $arguments = @(
        "-S", $SourceRoot,
        "-B", $build,
        "-G", "NMake Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $install)",
        "-Dblaspp_DIR=$configuredBlasppDir",
        "-DBUILD_TESTS=$(if ($BuildTests) { 'ON' } else { 'OFF' })"
    )
    if ($BuildTests) {
        $gtestPrefix = Require-EnvironmentVariable "googletest_PREFIX"
        $arguments += "-DCMAKE_PREFIX_PATH=$gtestPrefix"
    }
    if (-not $UseOpenMP) {
        $arguments += "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE"
    }
    if ($SanitizeAddress) {
        $arguments += "-DSANITIZE_ADDRESS=ON"
    }

    Invoke-Checked -Program "cmake" -Arguments $arguments
    Invoke-Checked -Program "cmake" -Arguments @(
        "--build", $build, "--target", "install"
    )
    return @{
        Build = $build
        Install = $install
    }
}

switch ($Task) {
    "CoreOpenMP" {
        $randblas = Install-RandBLAS `
            -Name "core-openmp" -BuildTests $true -UseOpenMP $true `
            -SanitizeAddress:$SanitizeAddress
        $env:OMP_NUM_THREADS = "1"
        Invoke-Checked -Program "ctest" -Arguments @(
            "--test-dir", $randblas.Build, "--output-on-failure"
        )
        $env:OMP_NUM_THREADS = "4"
        Invoke-Checked -Program "ctest" -Arguments @(
            "--test-dir", $randblas.Build, "--output-on-failure"
        )
    }

    "CoreSerial" {
        $randblas = Install-RandBLAS `
            -Name "core-serial" -BuildTests $true -UseOpenMP $false `
            -SanitizeAddress:$SanitizeAddress
        Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue
        Invoke-Checked -Program "ctest" -Arguments @(
            "--test-dir", $randblas.Build, "--output-on-failure"
        )
    }

    "Downstream" {
        $randblas = Install-RandBLAS `
            -Name "downstream-library" -BuildTests $false -UseOpenMP $true `
            -UseNativeDependencyPaths $true
        $build = Join-Path $WorkRoot "downstream-consumer-build"

        # Deliberately omit blaspp_DIR. This makes the smoke test exercise the
        # native-backslash dependency path recorded by RandBLASConfig.cmake
        # and guards its generated-path normalization.
        Invoke-Checked -Program "cmake" -Arguments @(
            "-S", (Join-Path $SourceRoot "test\downstream"),
            "-B", $build,
            "-G", "NMake Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_PREFIX_PATH=$(Convert-ToCMakePath $randblas.Install)"
        )
        Invoke-Checked -Program "cmake" -Arguments @(
            "--build", $build, "--target", "smoke"
        )
        Invoke-Checked -Program (Join-Path $build "smoke.exe")
    }

    "Examples" {
        $lapackppDir = Require-EnvironmentVariable "lapackpp_DIR"
        $randblas = Install-RandBLAS `
            -Name "examples-library" -BuildTests $false -UseOpenMP $true
        $build = Join-Path $WorkRoot "examples-build"
        $randblasDir = Join-Path $randblas.Install "lib\cmake\RandBLAS"

        # Pass external dependency directories explicitly here so the examples
        # job can probe example portability independently of the downstream
        # package-path smoke test.
        Invoke-Checked -Program "cmake" -Arguments @(
            "-S", (Join-Path $SourceRoot "examples"),
            "-B", $build,
            "-G", "NMake Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DRandBLAS_DIR=$(Convert-ToCMakePath $randblasDir)",
            "-Dblaspp_DIR=$blasppDir",
            "-Dlapackpp_DIR=$lapackppDir"
        )
        Invoke-Checked -Program "cmake" -Arguments @("--build", $build)
    }
}
