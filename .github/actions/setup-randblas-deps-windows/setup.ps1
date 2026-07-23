[CmdletBinding()]
param(
    [string] $DependencyRoot = "",
    [string] $VcpkgExecutable = "",
    [switch] $InstallLapackpp
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Program,

        [Parameter(Mandatory = $false)]
        [string[]] $Arguments = @()
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

function Find-PackageConfigDirectory {
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)][string] $ConfigName
    )

    $config = Get-ChildItem -LiteralPath $Root -Recurse -File -Filter $ConfigName |
        Select-Object -First 1
    if (-not $config) {
        throw "Could not find $ConfigName below $Root."
    }
    return $config.Directory.FullName
}

function Clone-Head {
    param(
        [Parameter(Mandatory = $true)][string] $Url,
        [Parameter(Mandatory = $true)][string] $Destination,
        [string] $Branch = ""
    )

    if (Test-Path -LiteralPath $Destination) {
        return
    }

    $arguments = @("clone", "--depth", "1")
    if ($Branch) {
        $arguments += @("--branch", $Branch)
    }
    $arguments += @($Url, $Destination)
    Invoke-Checked -Program "git" -Arguments $arguments
}

function Add-IncludeIfMissing {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Include,
        [Parameter(Mandatory = $true)][string] $After
    )

    $content = [IO.File]::ReadAllText($Path)
    if ($content.Contains($Include)) {
        return
    }
    if (-not $content.Contains($After)) {
        throw "Could not locate insertion point '$After' in $Path."
    }
    $content = $content.Replace($After, "$After`r`n$Include")
    [IO.File]::WriteAllText($Path, $content)
}

function Export-GitHubValue {
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [Parameter(Mandatory = $true)][string] $Value
    )

    Set-Item -Path "Env:$Name" -Value $Value
    if ($env:GITHUB_ENV) {
        "$Name=$Value" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
    }
    if ($env:GITHUB_OUTPUT) {
        "$($Name.ToLowerInvariant().Replace('_', '-'))=$Value" |
            Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    }
}

if (-not $DependencyRoot) {
    if ($env:GITHUB_WORKSPACE) {
        $DependencyRoot = Join-Path (Split-Path $env:GITHUB_WORKSPACE -Parent) "windows-deps"
    }
    else {
        throw "Pass -DependencyRoot when running setup.ps1 outside GitHub Actions."
    }
}

$DependencyRoot = [IO.Path]::GetFullPath($DependencyRoot)
if ($DependencyRoot -eq [IO.Path]::GetPathRoot($DependencyRoot)) {
    throw "DependencyRoot must not be a filesystem root."
}
New-Item -ItemType Directory -Force -Path $DependencyRoot | Out-Null

if (-not (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
    throw "MSVC is not active. Run from an x64 Native Tools prompt or initialize MSVC before invoking this script."
}

if (-not $VcpkgExecutable) {
    $vcpkgCandidates = @()
    if ($env:VCPKG_INSTALLATION_ROOT) {
        $vcpkgCandidates += Join-Path $env:VCPKG_INSTALLATION_ROOT "vcpkg.exe"
    }
    if ($env:VCPKG_ROOT) {
        $vcpkgCandidates += Join-Path $env:VCPKG_ROOT "vcpkg.exe"
    }
    $vcpkgCommand = Get-Command "vcpkg.exe" -ErrorAction SilentlyContinue
    if ($vcpkgCommand) {
        $vcpkgCandidates += $vcpkgCommand.Source
    }
    $VcpkgExecutable = $vcpkgCandidates |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}
if (-not $VcpkgExecutable) {
    throw "Could not find vcpkg.exe. Pass -VcpkgExecutable or set VCPKG_INSTALLATION_ROOT."
}

$vcpkgInstall = Join-Path $DependencyRoot "vcpkg-installed"
$mklRoot = Join-Path $vcpkgInstall "x64-windows"
$mklInclude = Join-Path $mklRoot "include"
$mklLib = Join-Path $mklRoot "lib"
$mklBin = Join-Path $mklRoot "bin"
$mklLibraries = @(
    (Join-Path $mklLib "mkl_intel_ilp64_dll.lib"),
    (Join-Path $mklLib "mkl_sequential_dll.lib"),
    (Join-Path $mklLib "mkl_core_dll.lib")
)

if (-not (Test-Path -LiteralPath $mklLibraries[0])) {
    Invoke-Checked -Program $VcpkgExecutable -Arguments @(
        "install",
        "intel-mkl:x64-windows",
        "--x-install-root=$vcpkgInstall"
    )
}
foreach ($path in @($mklInclude, $mklBin) + $mklLibraries) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Expected vcpkg oneMKL path does not exist: $path"
    }
}
$env:MKLROOT = $mklRoot
$env:PATH = "$mklBin;$env:PATH"

$gtestSource = Join-Path $DependencyRoot "googletest"
$gtestBuild = Join-Path $DependencyRoot "googletest-build"
$gtestInstall = Join-Path $DependencyRoot "googletest-install"
if (-not (Test-Path -LiteralPath (Join-Path $gtestInstall "lib\cmake\GTest\GTestConfig.cmake"))) {
    Clone-Head -Url "https://github.com/google/googletest.git" `
        -Destination $gtestSource -Branch "v1.17.0"
    Invoke-Checked -Program "cmake" -Arguments @(
        "-S", $gtestSource,
        "-B", $gtestBuild,
        "-G", "NMake Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $gtestInstall)",
        "-DBUILD_GMOCK=OFF",
        "-DINSTALL_GTEST=ON"
    )
    Invoke-Checked -Program "cmake" -Arguments @(
        "--build", $gtestBuild, "--target", "install"
    )
}

$random123Source = Join-Path $DependencyRoot "Random123"
$random123Install = Join-Path $DependencyRoot "Random123-install"
$random123Include = Join-Path $random123Install "include"
if (-not (Test-Path -LiteralPath (Join-Path $random123Include "Random123\philox.h"))) {
    Clone-Head -Url "https://github.com/DEShawResearch/Random123.git" `
        -Destination $random123Source
    New-Item -ItemType Directory -Force -Path $random123Include | Out-Null
    Copy-Item -LiteralPath (Join-Path $random123Source "include\Random123") `
        -Destination $random123Include -Recurse
}

$blasppSource = Join-Path $DependencyRoot "blaspp"
$blasppBuild = Join-Path $DependencyRoot "blaspp-build"
$blasppInstall = Join-Path $DependencyRoot "blaspp-install"
$blasppConfig = Get-ChildItem -LiteralPath $blasppInstall -Recurse -File `
    -Filter "blasppConfig.cmake" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $blasppConfig) {
    Clone-Head -Url "https://github.com/icl-utk-edu/blaspp.git" `
        -Destination $blasppSource
    $blasLibraryArgument = ($mklLibraries | ForEach-Object {
        Convert-ToCMakePath $_
    }) -join ";"
    Invoke-Checked -Program "cmake" -Arguments @(
        "-S", $blasppSource,
        "-B", $blasppBuild,
        "-G", "NMake Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $blasppInstall)",
        "-DBUILD_SHARED_LIBS=ON",
        "-Duse_cmake_find_blas=false",
        "-DBLAS_LIBRARIES=$blasLibraryArgument",
        "-Dblas_int=ilp64",
        "-Dblas_threaded=false",
        "-Duse_openmp=false",
        "-Dgpu_backend=none",
        "-Dbuild_tests=OFF"
    )
    Invoke-Checked -Program "cmake" -Arguments @(
        "--build", $blasppBuild, "--target", "install"
    )
}
$blasppDir = Find-PackageConfigDirectory `
    -Root $blasppInstall -ConfigName "blasppConfig.cmake"

$lapackppDir = ""
if ($InstallLapackpp) {
    $lapackppSource = Join-Path $DependencyRoot "lapackpp"
    $lapackppBuild = Join-Path $DependencyRoot "lapackpp-build"
    $lapackppInstall = Join-Path $DependencyRoot "lapackpp-install"
    $lapackppConfig = Get-ChildItem -LiteralPath $lapackppInstall -Recurse -File `
        -Filter "lapackppConfig.cmake" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $lapackppConfig) {
        Clone-Head -Url "https://github.com/icl-utk-edu/lapackpp.git" `
            -Destination $lapackppSource

        # Temporary upstream compatibility patches. These are no-ops once the
        # corresponding direct includes are present in LAPACK++ itself.
        Add-IncludeIfMissing `
            -Path (Join-Path $lapackppSource "include\lapack\config.h") `
            -Include "#include <stdint.h>" `
            -After '#include "lapack/defines.h"'
        Add-IncludeIfMissing `
            -Path (Join-Path $lapackppSource "src\lartg.cc") `
            -Include "#include <complex>" `
            -After '#include "lapack/fortran.h"'

        Invoke-Checked -Program "cmake" -Arguments @(
            "-S", $lapackppSource,
            "-B", $lapackppBuild,
            "-G", "NMake Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $lapackppInstall)",
            "-Dblaspp_DIR=$(Convert-ToCMakePath $blasppDir)",
            "-DBUILD_SHARED_LIBS=ON",
            "-Dgpu_backend=none",
            "-Dbuild_tests=OFF",
            "-DCMAKE_CXX_FLAGS=/EHsc /D__PRETTY_FUNCTION__=__FUNCSIG__"
        )
        Invoke-Checked -Program "cmake" -Arguments @(
            "--build", $lapackppBuild, "--target", "install"
        )
    }
    $lapackppDir = Find-PackageConfigDirectory `
        -Root $lapackppInstall -ConfigName "lapackppConfig.cmake"
}

$exports = [ordered]@{
    "blaspp_DIR" = Convert-ToCMakePath $blasppDir
    "Random123_DIR" = Convert-ToCMakePath $random123Include
    "googletest_PREFIX" = Convert-ToCMakePath $gtestInstall
    "MKLROOT" = Convert-ToCMakePath $mklRoot
}
if ($InstallLapackpp) {
    $exports["lapackpp_DIR"] = Convert-ToCMakePath $lapackppDir
}

foreach ($entry in $exports.GetEnumerator()) {
    Export-GitHubValue -Name $entry.Key -Value $entry.Value
    Write-Host "$($entry.Key)=$($entry.Value)"
}
if ($env:GITHUB_PATH) {
    (Convert-ToCMakePath $mklBin) |
        Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
}

# Stable output names consumed by action.yml.
if ($env:GITHUB_OUTPUT) {
    "blaspp-dir=$(Convert-ToCMakePath $blasppDir)" |
        Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    "random123-dir=$(Convert-ToCMakePath $random123Include)" |
        Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    "googletest-prefix=$(Convert-ToCMakePath $gtestInstall)" |
        Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    "mkl-root=$(Convert-ToCMakePath $mklRoot)" |
        Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    if ($InstallLapackpp) {
        "lapackpp-dir=$(Convert-ToCMakePath $lapackppDir)" |
            Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    }
}
