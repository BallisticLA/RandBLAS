# Toolchain architecture detection, shared by install/install.ps1 (user-facing
# preflight) and .github/actions/setup-randlapack-deps-windows/setup.ps1 (which
# also runs standalone in CI). Dot-source it; it defines functions only.
#
# Why this check exists: RandBLAS and every BLAS backend the installer
# provisions are 64-bit, but the "Developer PowerShell for VS" and "Developer
# Command Prompt for VS" Start-menu entries both default to an *x86* toolchain.
# An x86 linker cannot use an x64 import library, and the failure surfaces
# three layers down as BLAS++ reporting "BLAS library not found" -- which
# blames the libraries when the compiler is at fault. Note that the shell's own
# bitness is not a usable signal: the Developer Command Prompt is a 64-bit
# process that still selects x86 tools.

function Get-ClTargetArchitecture {
    # Returns the compiler's TARGET architecture, lowercased ("x64", "x86",
    # "arm64", "arm"), or "" if it genuinely cannot be determined.
    #
    # Three independent signals, most reliable first -- the same
    # probe-several-things approach Find-OneMklLayout uses, and for the same
    # reason: a missed detection here fails *open*, which defeats the check.
    #   1. VSCMD_ARG_TGT_ARCH, exported by vcvarsall.bat / VsDevCmd (and so
    #      by ilammy/msvc-dev-cmd in CI). Never localized.
    #   2. The toolset path: MSVC lays cl.exe out as
    #      ...\bin\Host<host>\<target>\cl.exe, a stable convention.
    #   3. The banner, last, for anything matching neither of the above.
    #      On its own this would be wrong on a localized Visual Studio, where
    #      the words around the architecture are translated.
    if ($env:VSCMD_ARG_TGT_ARCH) { return $env:VSCMD_ARG_TGT_ARCH.ToLowerInvariant() }
    $cl = Get-Command "cl.exe" -ErrorAction SilentlyContinue
    if (-not $cl) { return "" }
    if ($cl.Source -match '\\bin\\Host[^\\]+\\([^\\]+)\\cl\.exe$') {
        return $Matches[1].ToLowerInvariant()
    }
    # Native stderr merged via 2>&1 becomes ErrorRecords, which would throw
    # under $ErrorActionPreference = "Stop"; relax it for this one call.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $banner = (& $cl.Source 2>&1 | Out-String)
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($banner -match '\bfor\s+(x64|x86|ARM64|ARM)\b') { return $Matches[1].ToLowerInvariant() }
    return ""
}

function Get-ToolchainArchitectureProblem {
    # Returns a description of why $Arch is unusable, or "" if it is fine.
    # x86 and ARM64 fail for completely different reasons and deserve
    # different advice: x86 means the wrong shell was opened and is a
    # one-command fix, ARM64 means the platform is genuinely unsupported.
    param([string]$Arch)
    if ($Arch -eq "" -or $Arch -eq "x64" -or $Arch -eq "amd64") { return "" }
    if ($Arch -eq "x86") {
        # Single-quoted: the cmd one-liner contains both double quotes and
        # backticks, which are literal here but would need escaping in a
        # double-quoted PowerShell string.
        $vcvarsHint = 'for /f "usebackq delims=" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvars64.bat"'
        return ("cl.exe targets x86, but RandBLAS and its BLAS backends are 64-bit " +
            "(x64).`n" +
            "  You are in a 32-bit developer shell. 'Developer PowerShell for VS 2022' and " +
            "'Developer Command Prompt for VS 2022' both default to x86.`n" +
            "  Fix: open 'x64 Native Tools Command Prompt for VS 2022' from the Start menu, " +
            "or run this in any Command Prompt (any edition or version):`n" +
            "    $vcvarsHint`n" +
            "  Then delete the RandNLA-project directory before retrying: dependencies already " +
            "configured by the x86 compiler are reused as-is and would keep failing.")
    }
    return ("cl.exe targets $Arch, which this installer does not support: the Windows build " +
        "is x64-only.`n" +
        "  Intel oneMKL publishes no $Arch build, and the OpenBLAS binaries pinned here are " +
        "x64. Supplying an $Arch BLAS/LAPACK through -Backend custom is the only route, and " +
        "it is untested.`n" +
        "  If you meant to build x64, open 'x64 Native Tools Command Prompt for VS 2022'.")
}
