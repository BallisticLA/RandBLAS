[CmdletBinding()]
param(
    [string] $DependencyRoot = "",
    [string] $VcpkgExecutable = "",
    [switch] $InstallLapackpp,
    [switch] $SanitizeAddress
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

# Fetch exactly one commit or tag, and record where it came from.
#
# This replaces a clone that took a branch name and returned early whenever the
# destination merely existed. Two problems with that: a branch tip moves, so
# two runs of the same script could build different source; and reuse keyed on
# presence means changing a ref is a silent no-op for anyone who already has
# the directory, so the new pin never takes effect. The stamp is needed
# because a shallow fetch of a tag does not keep the tag ref locally, so git
# cannot be asked afterwards whether a tree is at the pin.
function Clone-Pinned {
    param(
        [Parameter(Mandatory = $true)][string] $Url,
        [Parameter(Mandatory = $true)][string] $Destination,
        [Parameter(Mandatory = $true)][string] $Ref
    )

    $stampPath = Join-Path $Destination ".randblas-provenance"
    $stamp = "$Url@$Ref"
    if ((Test-Path -LiteralPath $stampPath) -and
        ((Get-Content -LiteralPath $stampPath -Raw).Trim() -eq $stamp)) {
        Write-Host "Reusing $Destination (already at $Ref)"
        return
    }

    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -Recurse -Force -LiteralPath $Destination
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Invoke-Checked -Program "git" -Arguments @("-C", $Destination, "init", "--quiet")
    Invoke-Checked -Program "git" -Arguments @("-C", $Destination, "remote", "add", "origin", $Url)
    Invoke-Checked -Program "git" -Arguments @("-C", $Destination, "fetch", "--quiet", "--depth", "1", "origin", $Ref)
    Invoke-Checked -Program "git" -Arguments @("-C", $Destination, "checkout", "--quiet", "FETCH_HEAD")
    Set-Content -LiteralPath $stampPath -Value $stamp -Encoding ascii
}

#------------------------------------------------------------------ pins ------
# Immutable refs only: a tag or a full commit hash, never a branch. These match
# installers/install.sh and the refs RandLAPACK validated, so the two installers
# and CI cannot disagree about what they built.
#
# Pinned to the commits that merged new-Apple-Accelerate support on 2026-08-27
# (icl-utk-edu/blaspp#134, icl-utk-edu/lapackpp#88); they also contain the MSVC
# portability fixes (blaspp#132, lapackpp#87). The latest release of each,
# v2025.05.28, predates all of these.
$BlasppUrl    = "https://github.com/icl-utk-edu/blaspp.git"
$BlasppRef    = "2d8d4e937ac46fffab33d4174a4fc7659726dbda"
$LapackppUrl  = "https://github.com/icl-utk-edu/lapackpp.git"
$LapackppRef  = "b9439cf3c26d1655d88e7f510ae8b4f82fbeb687"
$Random123Url = "https://github.com/DEShawResearch/Random123.git"
$Random123Ref = "v1.14.0"
$GTestUrl     = "https://github.com/google/googletest.git"
$GTestRef     = "v1.18.0"

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
    if ($env:VSINSTALLDIR) {
        # Visual Studio 2022 17.6+ bundles vcpkg with the C++ workload; a
        # developer prompt exports VSINSTALLDIR but does not always put
        # vcpkg.exe on PATH.
        $vcpkgCandidates += Join-Path $env:VSINSTALLDIR "VC\vcpkg\vcpkg.exe"
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
    # Manifest mode is the only mode every vcpkg distribution supports: the
    # copy bundled with Visual Studio has no classic-mode instance, so
    # `vcpkg install intel-mkl:x64-windows` fails there outright. Generate a
    # minimal manifest and point every scratch tree at DependencyRoot -- the
    # bundled vcpkg lives under Program Files, where its default scratch
    # locations are not writable. The bundled vcpkg additionally requires
    # builtin-baseline; the pin below is vcpkg release 2026.07.29
    # (intel-mkl 2025.2.0), which also makes the oneMKL version independent
    # of the vcpkg copy's age.
    $vcpkgScratch = Join-Path $DependencyRoot "vcpkg-scratch"
    $manifestDir = Join-Path $vcpkgScratch "manifest"
    New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
    Set-Content -Path (Join-Path $manifestDir "vcpkg.json") -Encoding ascii -Value @(
        '{',
        '  "name": "randblas-windows-deps",',
        '  "version-string": "1",',
        '  "builtin-baseline": "9e593bb18ea69cc5095e012465dcd675a822ed0d",',
        '  "dependencies": [ "intel-mkl" ]',
        '}')
    Push-Location $manifestDir
    try {
        Invoke-Checked -Program $VcpkgExecutable -Arguments @(
            "install",
            "--triplet", "x64-windows",
            "--x-install-root=$vcpkgInstall",
            "--downloads-root=$(Join-Path $vcpkgScratch 'downloads')",
            "--x-buildtrees-root=$(Join-Path $vcpkgScratch 'buildtrees')",
            "--x-packages-root=$(Join-Path $vcpkgScratch 'packages')"
        )
    }
    finally {
        Pop-Location
    }
    # buildtrees/packages hold gigabytes of extracted installer scratch; the
    # installed prefix is self-contained. Keep downloads so a re-run (or a
    # CI downloads cache) skips the oneMKL fetch.
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
        (Join-Path $vcpkgScratch "buildtrees"), (Join-Path $vcpkgScratch "packages")
}
foreach ($path in @($mklInclude, $mklBin) + $mklLibraries) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Expected vcpkg oneMKL path does not exist: $path"
    }
}
$env:MKLROOT = $mklRoot
$env:PATH = "$mklBin;$env:PATH"

$gtestSource = Join-Path $DependencyRoot "googletest"
$gtestVariant = if ($SanitizeAddress) { "googletest-asan" } else { "googletest" }
$gtestBuild = Join-Path $DependencyRoot "$gtestVariant-build"
$gtestInstall = Join-Path $DependencyRoot "$gtestVariant-install"
if (-not (Test-Path -LiteralPath (Join-Path $gtestInstall "lib\cmake\GTest\GTestConfig.cmake"))) {
    Clone-Pinned -Url $GTestUrl -Destination $gtestSource -Ref $GTestRef
    $gtestArguments = @(
        "-S", $gtestSource,
        "-B", $gtestBuild,
        "-G", "NMake Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $gtestInstall)",
        "-DBUILD_GMOCK=OFF",
        "-DINSTALL_GTEST=ON"
    )
    if ($SanitizeAddress) {
        # MSVC's STL container-annotation mode must agree across every object
        # linked into a process. Build a separate instrumented GoogleTest
        # installation for the ASan job instead of disabling those checks.
        $gtestArguments += "-DCMAKE_CXX_FLAGS=/fsanitize=address /Zi"
    }
    Invoke-Checked -Program "cmake" -Arguments $gtestArguments
    Invoke-Checked -Program "cmake" -Arguments @(
        "--build", $gtestBuild, "--target", "install"
    )
}

$random123Source = Join-Path $DependencyRoot "Random123"
$random123Install = Join-Path $DependencyRoot "Random123-install"
$random123Include = Join-Path $random123Install "include"
if (-not (Test-Path -LiteralPath (Join-Path $random123Include "Random123\philox.h"))) {
    Clone-Pinned -Url $Random123Url -Destination $random123Source -Ref $Random123Ref
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
    Clone-Pinned -Url $BlasppUrl -Destination $blasppSource -Ref $BlasppRef
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
        Clone-Pinned -Url $LapackppUrl -Destination $lapackppSource -Ref $LapackppRef

        Invoke-Checked -Program "cmake" -Arguments @(
            "-S", $lapackppSource,
            "-B", $lapackppBuild,
            "-G", "NMake Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $lapackppInstall)",
            "-Dblaspp_DIR=$(Convert-ToCMakePath $blasppDir)",
            "-DBUILD_SHARED_LIBS=ON",
            "-Dgpu_backend=none",
            "-Dbuild_tests=OFF"
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
