#!/bin/bash
# RandBLAS autoinstaller for Linux and macOS.
#
# Builds RandBLAS and the dependencies it needs, into a self-contained
# "RandNLA-project" directory laid out as:
#   lib:     blaspp (and lapackpp, only if examples are requested) sources
#   install: RandBLAS-install, blaspp-install, Random123, googletest-install
#   build:   one build directory per project above
#
# Nothing is installed system-wide and your shell configuration is not touched.
#
# You bring a C++20 compiler, CMake 3.21+, Git and a BLAS. This script does not
# install compilers or package managers; when something is missing it says so
# and tells you the usual way to get it.
#
# Prerequisites and supported configurations are listed in INSTALL.md.

set -euo pipefail

usage() {
    # A heredoc rather than a line-range sed over this file's own comments:
    # the latter silently starts printing unrelated code the moment anyone
    # adds a line above it.
    cat <<'USAGE'
Usage: bash install/install.sh [options]

Backend selection:
  --blas=BACKEND        auto | openblas | mkl | accelerate | custom
                        (default: auto -- Accelerate on macOS, MKL on Linux
                        when MKLROOT is set, otherwise OpenBLAS)
  --blas-int=WIDTH      ilp64 | lp64. Default is ilp64 wherever the backend
                        can provide it, falling back to lp64 with a warning.
                        Accelerate is lp64-only and rejects ilp64.
  --blas-libraries=L    Link line for --blas=custom, e.g.
                        "/opt/aocl/lib/libblis.so;/opt/aocl/lib/libflame.so"

Locations:
  --project-dir=DIR     Where dependencies, builds and installs go.
                        Default: $RANDNLA_PROJECT_DIR if set, otherwise
                        ../RandNLA-project next to this clone.
  --prefix=DIR          Install RandBLAS itself here instead of
                        <project-dir>/install/RandBLAS-install. Dependencies
                        still go in the project directory.

Build:
  -j, --jobs N          Parallel build jobs (default: number of cores)
      --fresh           Clear build directories and rebuild dependencies
      --no-tests        Do not provision GoogleTest, and configure with
                        -DBUILD_TESTS=OFF
      --no-openmp       Configure without OpenMP
      --examples        Build the examples too, instead of offering them
                        after the install finishes

Output:
  -y, --yes             Assume "yes" at every prompt. This is also the
                        behavior when stdin is not a terminal (CI, pipes).
      --no-progress     Plain one-line-per-step output, no redrawing
  -h, --help            Show this help and exit

Every option has an environment-variable equivalent (flags win):
  RANDBLAS_INSTALL_BLAS, RANDBLAS_INSTALL_BLAS_INT,
  RANDBLAS_INSTALL_BLAS_LIBRARIES, RANDBLAS_INSTALL_PROJECT_DIR,
  RANDBLAS_INSTALL_PREFIX, RANDBLAS_INSTALL_JOBS, RANDBLAS_INSTALL_FRESH,
  RANDBLAS_INSTALL_TESTS, RANDBLAS_INSTALL_OPENMP,
  RANDBLAS_INSTALL_EXAMPLES, RANDBLAS_INSTALL_YES,
  RANDBLAS_INSTALL_PROGRESS

Already-installed dependencies are reused when pointed at by:
  BLASPP_INSTALL_DIR, RANDOM123_INSTALL_DIR, LAPACKPP_INSTALL_DIR, GTEST_ROOT

All compiler output goes to <project-dir>/install.log; the console shows one
line per step. On failure the log path is printed.
USAGE
}

#==============================================================================
# Option parsing. Environment variables provide defaults; flags override.
#==============================================================================
BLAS_BACKEND="${RANDBLAS_INSTALL_BLAS:-auto}"
BLAS_INT_CHOICE="${RANDBLAS_INSTALL_BLAS_INT:-auto}"   # auto | ilp64 | lp64
BLAS_LIBRARIES_ARG="${RANDBLAS_INSTALL_BLAS_LIBRARIES:-}"
PROJECT_DIR_OVERRIDE="${RANDBLAS_INSTALL_PROJECT_DIR:-}"
PREFIX_OVERRIDE="${RANDBLAS_INSTALL_PREFIX:-}"
JOBS="${RANDBLAS_INSTALL_JOBS:-}"
FRESH="${RANDBLAS_INSTALL_FRESH:-0}"
WANT_TESTS="${RANDBLAS_INSTALL_TESTS:-1}"
WANT_OPENMP="${RANDBLAS_INSTALL_OPENMP:-1}"
WANT_EXAMPLES="${RANDBLAS_INSTALL_EXAMPLES:-0}"
ASSUME_YES="${RANDBLAS_INSTALL_YES:-0}"
WANT_PROGRESS="${RANDBLAS_INSTALL_PROGRESS:-1}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --blas)             BLAS_BACKEND="${2:?--blas requires a backend}"; shift ;;
        --blas=*)           BLAS_BACKEND="${1#*=}" ;;
        --blas-int)         BLAS_INT_CHOICE="${2:?--blas-int requires a width}"; shift ;;
        --blas-int=*)       BLAS_INT_CHOICE="${1#*=}" ;;
        --blas-libraries)   BLAS_LIBRARIES_ARG="${2:?--blas-libraries requires a value}"; shift ;;
        --blas-libraries=*) BLAS_LIBRARIES_ARG="${1#*=}" ;;
        --project-dir)      PROJECT_DIR_OVERRIDE="${2:?--project-dir requires a path}"; shift ;;
        --project-dir=*)    PROJECT_DIR_OVERRIDE="${1#*=}" ;;
        --prefix)           PREFIX_OVERRIDE="${2:?--prefix requires a path}"; shift ;;
        --prefix=*)         PREFIX_OVERRIDE="${1#*=}" ;;
        -j|--jobs)          JOBS="${2:?--jobs requires a number}"; shift ;;
        --jobs=*)           JOBS="${1#*=}" ;;
        -j*)                JOBS="${1#-j}" ;;   # attached form, as in -j8
        --fresh)            FRESH=1 ;;
        --no-tests)         WANT_TESTS=0 ;;
        --no-openmp)        WANT_OPENMP=0 ;;
        --examples)         WANT_EXAMPLES=1 ;;
        -y|--yes)           ASSUME_YES=1 ;;
        --no-progress)      WANT_PROGRESS=0 ;;
        -h|--help)          usage; exit 0 ;;
        *) printf 'Unknown option: %s (see --help)\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

case "$BLAS_BACKEND" in
    auto|openblas|mkl|accelerate|custom) ;;
    *) die "--blas must be auto, openblas, mkl, accelerate or custom (got '$BLAS_BACKEND')" ;;
esac
case "$BLAS_INT_CHOICE" in
    auto|ilp64|lp64) ;;
    *) die "--blas-int must be ilp64 or lp64 (got '$BLAS_INT_CHOICE')" ;;
esac
if [[ "$BLAS_BACKEND" == "custom" && -z "$BLAS_LIBRARIES_ARG" ]]; then
    die "--blas=custom needs --blas-libraries=<semicolon-separated link line>"
fi
if [[ -n "$BLAS_LIBRARIES_ARG" && "$BLAS_BACKEND" != "custom" ]]; then
    die "--blas-libraries only applies to --blas=custom (backend is '$BLAS_BACKEND')"
fi

if [[ -z "$JOBS" ]]; then
    JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
fi

#==============================================================================
# Interactivity and output style.
#
# Prompts happen only on a terminal and only without --yes. When stdin is not a
# terminal (piped, CI) every prompt silently takes its default, so this script
# can never hang waiting for input nobody is there to give.
#==============================================================================
INTERACTIVE=0
if [[ -t 0 && "$ASSUME_YES" != "1" ]]; then
    INTERACTIVE=1
fi

# ask <question> <default y|n> -> returns 0 for yes.
ask() {
    local question="$1" default="$2" reply
    if [[ "$INTERACTIVE" != "1" ]]; then
        [[ "$default" == "y" ]]
        return
    fi
    read -r -p "$question [$( [[ $default == y ]] && echo Y/n || echo y/N )]: " reply
    reply="${reply:-$default}"
    [[ "$reply" == "y" || "$reply" == "Y" || "$reply" == "yes" ]]
}

# Plain output when stdout is not a terminal, or when NO_COLOR / TERM=dumb ask
# for it. Piped output must stay free of escape sequences so that build logs
# and CI transcripts remain readable.
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" && "$WANT_PROGRESS" == "1" ]]; then
    C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_WARN=$'\033[33m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
    C_OK=""; C_ERR=""; C_WARN=""; C_BOLD=""; C_OFF=""
fi

note() { printf '%s\n' "$*"; }
warn() { printf '%swarning:%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }

# Collected and reprinted in the final summary. A warning emitted twenty
# minutes and several thousand log lines before the summary is a warning
# nobody reads.
WARNINGS=()
record_warning() { WARNINGS+=("$1"); warn "$1"; }

#==============================================================================
# Toolchain preflight. Report everything missing at once rather than failing on
# the first one, so a bare machine takes one round trip instead of three.
#==============================================================================
UNAME_S="$(uname -s)"
MISSING=()
command -v cmake >/dev/null 2>&1 || MISSING+=("cmake")
command -v git   >/dev/null 2>&1 || MISSING+=("git")
if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && \
   ! command -v clang++ >/dev/null 2>&1; then
    MISSING+=("a C++ compiler")
fi
if (( ${#MISSING[@]} )); then
    printf 'ERROR: missing prerequisites: %s\n\n' "${MISSING[*]}" >&2
    if [[ "$UNAME_S" == "Darwin" ]]; then
        printf '  xcode-select --install    # Apple Clang and git\n' >&2
        printf '  brew install cmake\n\n' >&2
    else
        printf '  sudo apt install g++ cmake git      # Debian, Ubuntu\n' >&2
        printf '  sudo dnf install gcc-c++ cmake git  # Fedora, RHEL\n\n' >&2
    fi
    printf 'See INSTALL.md for the full prerequisite list.\n' >&2
    exit 1
fi

CMAKE_VERSION="$(cmake --version | head -n1 | awk '{print $3}')"
if [[ "$(printf '%s\n3.21\n' "$CMAKE_VERSION" | sort -V | head -n1)" != "3.21" ]]; then
    die "CMake 3.21 or later is required (found $CMAKE_VERSION). See INSTALL.md."
fi

# RandBLAS uses C++20 concepts, which GCC only implements completely from 13.
if command -v g++ >/dev/null 2>&1 && [[ "${CXX:-g++}" != *clang* ]]; then
    GXX_MAJOR="$(g++ -dumpversion 2>/dev/null | cut -d. -f1)"
    if [[ -n "$GXX_MAJOR" && "$GXX_MAJOR" -lt 13 ]]; then
        record_warning "g++ $GXX_MAJOR is older than the supported minimum of 13; RandBLAS uses C++20 concepts and may not compile."
    fi
fi

#==============================================================================
# Project layout.
#
# Precedence for the layout root: --project-dir, then RANDNLA_PROJECT_DIR, then
# a sibling of this clone. Honouring RANDNLA_PROJECT_DIR is what lets this
# installer and RandLAPACK's share one dependency tree: whichever runs second
# finds the first one's BLAS++ and reuses it.
#
# Unlike RandLAPACK's installer, this one never moves your clone. Relocating a
# repository out from under the user breaks git worktrees and is surprising;
# the layout below is created by mkdir regardless of where the clone lives, and
# lib/RandBLAS is a symlink so the tree still reads as complete.
#==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -n "$PROJECT_DIR_OVERRIDE" ]]; then
    PROJECT_DIR="$PROJECT_DIR_OVERRIDE"
elif [[ -n "${RANDNLA_PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="$RANDNLA_PROJECT_DIR"
else
    PROJECT_DIR="$(dirname "$REPO_DIR")/RandNLA-project"
fi
mkdir -p "$PROJECT_DIR"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

mkdir -p "$PROJECT_DIR"/{install,lib,build}

# A symlink, not a move: the clone stays where the user put it.
if [[ ! -e "$PROJECT_DIR/lib/RandBLAS" ]]; then
    ln -s "$REPO_DIR" "$PROJECT_DIR/lib/RandBLAS"
fi

RANDBLAS_INSTALL_DIR="${PREFIX_OVERRIDE:-$PROJECT_DIR/install/RandBLAS-install}"

LOG="$PROJECT_DIR/install.log"
# Appended, not truncated: the previous run's output is exactly what you want
# when the current run fails the same way.
{
    printf '\n===============================================================\n'
    printf 'RandBLAS install started %s\n' "$(date)"
    printf 'command: %s\n' "$0 $*"
    printf '===============================================================\n'
} >> "$LOG"

STEP=0
TOTAL_STEPS=0
run_step() {
    local label="$1"; shift
    STEP=$((STEP + 1))
    printf '%s[%d/%d]%s %s ... ' "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF" "$label"
    local t0 t1
    t0=$(date +%s)
    {
        printf '\n===== [%d/%d] %s =====\n' "$STEP" "$TOTAL_STEPS" "$label"
        printf '$ %s\n' "$*"
    } >> "$LOG"
    if "$@" >> "$LOG" 2>&1; then
        t1=$(date +%s)
        printf '%sdone%s (%ds)\n' "$C_OK" "$C_OFF" "$((t1 - t0))"
    else
        printf '%sFAILED%s\n' "$C_ERR" "$C_OFF" >&2
        printf '\nStep "%s" failed. Full output: %s\n' "$label" "$LOG" >&2
        printf 'The last 20 log lines:\n' >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
}

skip_step() {
    STEP=$((STEP + 1))
    printf '%s[%d/%d]%s %s\n' "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF" "$1"
}

#==============================================================================
# Provenance stamps.
#
# A dependency is reused only when it was built from the source we would build
# from now, and in the configuration we want now. Reuse keyed on mere presence
# means that changing a pin, or switching BLAS backend, is a silent no-op for
# anyone who already ran this script -- they keep the old artifact and the new
# setting never takes effect.
#==============================================================================
stamp_file() { printf '%s/.randblas-provenance' "$1"; }

stamp_matches() {  # <install-dir> <expected-stamp>
    local dir="$1" expected="$2"
    [[ -f "$(stamp_file "$dir")" ]] || return 1
    [[ "$(cat "$(stamp_file "$dir")")" == "$expected" ]]
}

write_stamp() {  # <install-dir> <stamp>
    printf '%s\n' "$2" > "$(stamp_file "$1")"
}

# Shallow-fetch exactly one commit or tag, so the pin cannot drift the way a
# branch name would and we do not pay for the full history. The source tree
# gets its own stamp: a shallow checkout of a tag does not keep the tag ref
# locally, so "is this clone at the pinned ref?" cannot be answered by asking
# git afterwards, and without the stamp a source tree left over from an older
# pin would be reused as if it were current.
clone_pinned() {  # <url> <dest> <ref>
    local url="$1" dest="$2" ref="$3"
    rm -rf "$dest"
    mkdir -p "$dest"
    git -C "$dest" init --quiet
    git -C "$dest" remote add origin "$url"
    git -C "$dest" fetch --quiet --depth 1 origin "$ref"
    git -C "$dest" checkout --quiet FETCH_HEAD
    write_stamp "$dest" "$url@$ref"
}

source_is_current() {  # <dest> <url> <ref>
    stamp_matches "$1" "$2@$3"
}

#==============================================================================
# Dependency pins. Immutable refs only: a tag or a full commit hash, never a
# branch name. These match the refs RandLAPACK's Windows provisioner validated.
#==============================================================================
BLASPP_URL="https://github.com/icl-utk-edu/blaspp.git"
# The commit that merged the MSVC portability fix (blaspp PR #132, 2026-08-06).
# Not in a release yet -- the latest tag, v2025.05.28, predates it. Move to a
# tag once one includes it.
BLASPP_REF="30571853f980d3a2a1737124ea4789e025a5e045"

LAPACKPP_URL="https://github.com/icl-utk-edu/lapackpp.git"
LAPACKPP_REF="40b9d0daf29b6f1f3fa58bc3f22bd6cfb2c67fe4"

RANDOM123_URL="https://github.com/DEShawResearch/Random123.git"
RANDOM123_REF="v1.14.0"

GTEST_URL="https://github.com/google/googletest.git"
GTEST_REF="v1.18.0"

#==============================================================================
# Backend resolution.
#
# "auto" prefers what the platform actually ships: Accelerate on macOS, MKL on
# Linux when MKLROOT says it is installed, OpenBLAS otherwise. The choice is
# always printed, because a silent default is the thing people later cannot
# explain.
#==============================================================================
BREW_PREFIX=""
if [[ "$UNAME_S" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    # Never hardcode /opt/homebrew: that is Apple Silicon only, and breaks both
    # Intel Macs (/usr/local) and any custom HOMEBREW_PREFIX.
    BREW_PREFIX="$(brew --prefix)"
fi

if [[ "$BLAS_BACKEND" == "auto" ]]; then
    if [[ "$UNAME_S" == "Darwin" ]]; then
        BLAS_BACKEND="accelerate"
    elif [[ -n "${MKLROOT:-}" && -d "${MKLROOT:-}" ]]; then
        BLAS_BACKEND="mkl"
    else
        BLAS_BACKEND="openblas"
    fi
    note "Selected BLAS backend: $BLAS_BACKEND (from --blas=auto)"
fi

# Integer width policy: prefer ILP64 wherever the backend can provide it, and
# fall back to LP64 only where it cannot.
#
# ILP64 matters because LP64 caps every individual BLAS dimension at 2^31, and
# because RandBLAS's MKL sparse path requires MKL_INT to match its int64_t
# sparse indices. RandBLAS's own API is int64_t either way -- BLAS++ presents
# int64_t regardless of the underlying width -- so this choice is about what
# the BLAS underneath can represent, not about RandBLAS's interface.
#
# Accelerate is the one backend with no ILP64 route at all. Apple has shipped
# an ILP64 interface since macOS 13.3, behind ACCELERATE_NEW_LAPACK and
# ACCELERATE_LAPACK_ILP64, but BLAS++ does not implement it: BLASFinder.cmake
# emits only "-framework Accelerate", the legacy LP64 path. Tracked upstream as
# icl-utk-edu/lapackpp#43.
WIDTH_ORDER=()
case "$BLAS_BACKEND" in
    accelerate)
        if [[ "$BLAS_INT_CHOICE" == "ilp64" ]]; then
            die "--blas-int=ilp64 is not available with Accelerate: BLAS++ implements only Apple's legacy LP64 interface (see icl-utk-edu/lapackpp#43). Use --blas=openblas or --blas=mkl for ILP64."
        fi
        WIDTH_ORDER=(int32)
        ;;
    *)
        case "$BLAS_INT_CHOICE" in
            ilp64) WIDTH_ORDER=(int64) ;;
            lp64)  WIDTH_ORDER=(int32) ;;
            auto)  WIDTH_ORDER=(int64 int32) ;;   # ILP64 first, LP64 as fallback
        esac
        ;;
esac

# blaspp's own backend selector. Its matcher accepts "apple" or "accelerate".
BLASPP_BACKEND_FLAGS=()
case "$BLAS_BACKEND" in
    openblas)   BLASPP_BACKEND_FLAGS=(-Dblas=openblas) ;;
    mkl)        BLASPP_BACKEND_FLAGS=(-Dblas=mkl) ;;
    accelerate) BLASPP_BACKEND_FLAGS=(-Dblas=apple) ;;
    custom)     BLASPP_BACKEND_FLAGS=(-DBLAS_LIBRARIES="$BLAS_LIBRARIES_ARG") ;;
esac

#==============================================================================
# OpenMP.
#
# Apple Clang ships no OpenMP runtime. Homebrew's libomp supplies one, but only
# via -Xpreprocessor -fopenmp plus explicit library paths, so it has to be wired
# up by hand rather than found by FindOpenMP.
#==============================================================================
OPENMP_FLAGS=()
if [[ "$WANT_OPENMP" != "1" ]]; then
    OPENMP_FLAGS=(-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE)
elif [[ "$UNAME_S" == "Darwin" ]]; then
    LIBOMP=""
    if [[ -n "$BREW_PREFIX" && -f "$BREW_PREFIX/opt/libomp/lib/libomp.dylib" ]]; then
        LIBOMP="$BREW_PREFIX/opt/libomp"
    fi
    if [[ -n "$LIBOMP" ]]; then
        export CXXFLAGS="${CXXFLAGS:-} -Xpreprocessor -fopenmp -I$LIBOMP/include"
        export LDFLAGS="${LDFLAGS:-} -L$LIBOMP/lib"
        OPENMP_FLAGS=(
            "-DOpenMP_CXX_LIB_NAMES=omp"
            "-DOpenMP_omp_LIBRARY=$LIBOMP/lib/libomp.dylib"
            "-DOpenMP_CXX_FLAGS=-Xpreprocessor;-fopenmp"
        )
    else
        OPENMP_FLAGS=(-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE)
        record_warning "OpenMP is unavailable, so RandBLAS will be single-threaded. Apple Clang has no OpenMP runtime; install one with 'brew install libomp' and re-run."
        WANT_OPENMP=0
    fi
fi

#==============================================================================
# Dependency discovery. An install pointed at by an environment variable is
# taken as given: the user knows something we do not, and rebuilding over it
# would be both slow and rude.
#==============================================================================
find_cmake_config() {  # <root> <package> -> prints the config dir, or nothing
    local root="$1" pkg="$2" libdir
    for libdir in lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu; do
        if [[ -f "$root/$libdir/cmake/$pkg/${pkg}Config.cmake" ]]; then
            printf '%s/%s/cmake/%s' "$root" "$libdir" "$pkg"
            return 0
        fi
    done
    return 0
}

# blaspp installs are per-configuration: an ILP64 MKL build and an LP64
# OpenBLAS build cannot share a directory, and silently reusing one for the
# other is exactly the mismatch this installer exists to prevent.
BLASPP_INSTALL="$PROJECT_DIR/install/blaspp-$BLAS_BACKEND-install"
RANDOM123_INSTALL="$PROJECT_DIR/install/Random123-install"
GTEST_INSTALL="$PROJECT_DIR/install/googletest-install"
LAPACKPP_INSTALL="$PROJECT_DIR/install/lapackpp-install"

EXTERNAL_BLASPP=0
EXTERNAL_RANDOM123=0
EXTERNAL_GTEST=0

BLASPP_CMAKE_DIR=""
RANDOM123_DIR=""
GTEST_ROOT_DIR=""

note ""
note "Dependency discovery:"

if [[ -n "${BLASPP_INSTALL_DIR:-}" ]]; then
    BLASPP_CMAKE_DIR="$(find_cmake_config "$BLASPP_INSTALL_DIR" blaspp)"
    if [[ -n "$BLASPP_CMAKE_DIR" ]]; then
        EXTERNAL_BLASPP=1
        note "  [blaspp]    external install: $BLASPP_INSTALL_DIR"
    else
        note "  [blaspp]    BLASPP_INSTALL_DIR is set but holds no blasppConfig.cmake; building from source."
    fi
fi
if [[ -n "${RANDOM123_INSTALL_DIR:-}" && -f "$RANDOM123_INSTALL_DIR/include/Random123/philox.h" ]]; then
    EXTERNAL_RANDOM123=1
    RANDOM123_DIR="$RANDOM123_INSTALL_DIR/include"
    note "  [Random123] external install: $RANDOM123_INSTALL_DIR"
fi
if [[ -n "${GTEST_ROOT:-}" && -f "$GTEST_ROOT/include/gtest/gtest.h" ]]; then
    EXTERNAL_GTEST=1
    GTEST_ROOT_DIR="$GTEST_ROOT"
    note "  [GoogleTest] external install: $GTEST_ROOT"
fi

# What we would build, and therefore what a prior install must match to be
# reused. Backend and integer width are part of the stamp precisely because
# the directory name alone cannot distinguish an ILP64 build from an LP64 one.
BLASPP_STAMP_BASE="$BLASPP_URL@$BLASPP_REF backend=$BLAS_BACKEND libs=$BLAS_LIBRARIES_ARG"
RANDOM123_STAMP="$RANDOM123_URL@$RANDOM123_REF"
GTEST_STAMP="$GTEST_URL@$GTEST_REF"

#==============================================================================
# Step accounting. Computed up front so "[3/7]" means something.
#==============================================================================
BUILD_BLASPP=0
BUILD_RANDOM123=0
BUILD_GTEST=0

if (( ! EXTERNAL_BLASPP )); then BUILD_BLASPP=1; fi
if (( ! EXTERNAL_RANDOM123 )); then BUILD_RANDOM123=1; fi
if (( WANT_TESTS && ! EXTERNAL_GTEST )); then BUILD_GTEST=1; fi

if (( FRESH )); then
    rm -rf "$PROJECT_DIR/build"
    mkdir -p "$PROJECT_DIR/build"
fi

# Per dependency: BLAS++ spends three steps (source, configure, build), which
# the reuse path also spends as skips so the counter agrees; Random123 one;
# GoogleTest two. Then RandBLAS configure, RandBLAS build, and verification.
# Examples add lapackpp source, lapackpp build, examples configure, examples
# build. Keep these in step with the run_step calls below -- a counter that
# reads "[9/7]" is worse than no counter.
TOTAL_STEPS=$(( BUILD_BLASPP * 3 + BUILD_RANDOM123 + BUILD_GTEST * 2 + 3 ))
if (( WANT_EXAMPLES )); then TOTAL_STEPS=$(( TOTAL_STEPS + 4 )); fi
note ""
#==============================================================================
# BLAS++.
#
# The integer width is settled here, by trying to configure BLAS++ at each
# width in WIDTH_ORDER until one succeeds. Letting BLAS++ do the searching is
# deliberate: it already knows the library names, symbol suffixes and probe
# programs for every vendor, and a second implementation here would be a worse
# copy that drifts.
#
# Each attempt gets a clean build directory. BLAS++ caches its detection
# results, and re-running cmake over a directory where detection previously
# failed regenerates blas/defines.h without the Fortran-mangling and backend
# defines, which then breaks every downstream compile in a way that looks
# nothing like the original failure.
#==============================================================================
BLAS_INT_RESOLVED=""
BLASPP_SRC="$PROJECT_DIR/lib/blaspp"

configure_blaspp_at_width() {  # <width int32|int64> -> 0 on success
    local width="$1"
    local build="$PROJECT_DIR/build/blaspp-build-$width"
    rm -rf "$build"
    mkdir -p "$build"
    cmake -S "$BLASPP_SRC" -B "$build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$BLASPP_INSTALL" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
        -Dblas_int="$width" \
        -Dbuild_tests=OFF \
        "${BLASPP_BACKEND_FLAGS[@]}" \
        "${OPENMP_FLAGS[@]}" >> "$LOG" 2>&1
}

if (( BUILD_BLASPP )); then
    # Reuse only when the recorded provenance matches what we would build now,
    # including the resolved width, which is stored beside the stamp.
    PRIOR_WIDTH="$(cat "$BLASPP_INSTALL/.randblas-width" 2>/dev/null || true)"
    REUSABLE=0
    if (( ! FRESH )) && [[ -n "$PRIOR_WIDTH" ]] && \
       stamp_matches "$BLASPP_INSTALL" "$BLASPP_STAMP_BASE width=$PRIOR_WIDTH"; then
        # A prior build is only reusable if its width is one we would accept.
        for w in "${WIDTH_ORDER[@]}"; do
            if [[ "$w" == "$PRIOR_WIDTH" ]]; then REUSABLE=1; break; fi
        done
        if (( ! REUSABLE )); then
            note "  [blaspp] existing install is $PRIOR_WIDTH but this run wants ${WIDTH_ORDER[0]}; rebuilding."
        fi
    fi

    if (( REUSABLE )); then
        BLAS_INT_RESOLVED="$PRIOR_WIDTH"
        BLASPP_CMAKE_DIR="$(find_cmake_config "$BLASPP_INSTALL" blaspp)"
        # Three skips, matching the three steps the build path below spends, so
        # the step counter reads the same either way.
        skip_step "BLAS++ source ... already present"
        skip_step "BLAS++ ... reusing the $BLAS_INT_RESOLVED install"
        skip_step "BLAS++ ... already built"
    else
        if source_is_current "$BLASPP_SRC" "$BLASPP_URL" "$BLASPP_REF"; then
            skip_step "BLAS++ source ... already at $BLASPP_REF"
        else
            run_step "Fetching BLAS++ ($BLASPP_REF)" \
                clone_pinned "$BLASPP_URL" "$BLASPP_SRC" "$BLASPP_REF"
        fi

        STEP=$((STEP + 1))
        printf '%s[%d/%d]%s Configuring BLAS++ ... ' "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$C_OFF"
        for width in "${WIDTH_ORDER[@]}"; do
            if configure_blaspp_at_width "$width"; then
                BLAS_INT_RESOLVED="$width"
                break
            fi
        done
        if [[ -z "$BLAS_INT_RESOLVED" ]]; then
            printf '%sFAILED%s\n' "$C_ERR" "$C_OFF" >&2
            printf '\nBLAS++ could not find a usable %s BLAS at any of: %s\n' \
                "$BLAS_BACKEND" "${WIDTH_ORDER[*]}" >&2
            printf 'Full output: %s\n' "$LOG" >&2
            case "$BLAS_BACKEND" in
                openblas) printf '\n  sudo apt install libopenblas64-dev   # ILP64, Debian/Ubuntu\n  sudo apt install libopenblas-dev     # LP64\n' >&2 ;;
                mkl)      printf '\n  Set MKLROOT, or source the oneAPI setvars script.\n' >&2 ;;
            esac
            exit 1
        fi
        printf '%sdone%s (requested %s)\n' "$C_OK" "$C_OFF" "$BLAS_INT_RESOLVED"

        run_step "Building and installing BLAS++" \
            cmake --build "$PROJECT_DIR/build/blaspp-build-$BLAS_INT_RESOLVED" \
                -j "$JOBS" --target install

        # The width actually built, read back from BLAS++'s own generated
        # header rather than inferred from which configure attempt succeeded.
        #
        # Those two can differ. BLAS++ probes int32 first and int64 second
        # (BLASFinder.cmake), while blas_int only filters which *library names*
        # to consider. For MKL that is enough, because mkl_intel_lp64 and
        # mkl_intel_ilp64 are different libraries. For OpenBLAS there is only
        # -lopenblas, so asking for int64 and getting a configure success tells
        # us nothing -- an LP64 OpenBLAS passes the int32 probe and is accepted.
        # Trusting the request here would stamp an LP64 install as ILP64, and
        # every later run would reuse it believing it had ILP64.
        if grep -q '^#define BLAS_ILP64' "$BLASPP_INSTALL/include/blas/defines.h" 2>/dev/null; then
            BLAS_INT_BUILT="int64"
        else
            BLAS_INT_BUILT="int32"
        fi
        if [[ "$BLAS_INT_BUILT" != "$BLAS_INT_RESOLVED" ]]; then
            note "  [blaspp] requested $BLAS_INT_RESOLVED, BLAS++ selected $BLAS_INT_BUILT"
        fi
        BLAS_INT_RESOLVED="$BLAS_INT_BUILT"

        if [[ "$BLAS_INT_RESOLVED" == "int32" && "${WIDTH_ORDER[0]}" == "int64" ]]; then
            record_warning "No ILP64 $BLAS_BACKEND was available, so BLAS++ was built LP64 (32-bit BLAS integers). RandBLAS works either way, but individual BLAS dimensions are then capped at 2^31 and the MKL sparse path is unavailable. For an ILP64 OpenBLAS, install one (on Debian or Ubuntu: libopenblas64-dev) and pass its library explicitly with --blas=custom --blas-libraries=... --blas-int=ilp64, because BLAS++ only ever looks for plain -lopenblas."
        fi

        write_stamp "$BLASPP_INSTALL" "$BLASPP_STAMP_BASE width=$BLAS_INT_RESOLVED"
        printf '%s\n' "$BLAS_INT_RESOLVED" > "$BLASPP_INSTALL/.randblas-width"
        BLASPP_CMAKE_DIR="$(find_cmake_config "$BLASPP_INSTALL" blaspp)"
    fi
fi

[[ -n "$BLASPP_CMAKE_DIR" ]] || die "BLAS++ was installed but blasppConfig.cmake could not be located under $BLASPP_INSTALL"

#==============================================================================
# Random123. Header-only: fetch and copy, nothing to build.
#==============================================================================
if (( BUILD_RANDOM123 )); then
    if (( ! FRESH )) && stamp_matches "$RANDOM123_INSTALL" "$RANDOM123_STAMP"; then
        RANDOM123_DIR="$RANDOM123_INSTALL/include"
        skip_step "Random123 ... reusing existing install"
    else
        install_random123() {
            local src="$PROJECT_DIR/lib/Random123"
            clone_pinned "$RANDOM123_URL" "$src" "$RANDOM123_REF"
            rm -rf "$RANDOM123_INSTALL/include/Random123"
            mkdir -p "$RANDOM123_INSTALL/include"
            cp -r "$src/include/Random123" "$RANDOM123_INSTALL/include/Random123"
        }
        run_step "Fetching Random123 ($RANDOM123_REF)" install_random123
        write_stamp "$RANDOM123_INSTALL" "$RANDOM123_STAMP"
        RANDOM123_DIR="$RANDOM123_INSTALL/include"
    fi
fi

#==============================================================================
# GoogleTest.
#
# Provisioned rather than assumed: RandBLAS defaults BUILD_TESTS to ON while
# its find_package(GTest) is not REQUIRED, so a machine without GoogleTest
# produces a build with zero tests that otherwise looks like a success.
#==============================================================================
if (( BUILD_GTEST )); then
    if (( ! FRESH )) && stamp_matches "$GTEST_INSTALL" "$GTEST_STAMP"; then
        GTEST_ROOT_DIR="$GTEST_INSTALL"
        skip_step "GoogleTest ... reusing existing install"
        skip_step "GoogleTest ... already built"
    else
        GTEST_SRC="$PROJECT_DIR/lib/googletest"
        run_step "Fetching GoogleTest ($GTEST_REF)" \
            clone_pinned "$GTEST_URL" "$GTEST_SRC" "$GTEST_REF"
        run_step "Building and installing GoogleTest" \
            bash -c 'cmake -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release \
                        -DCMAKE_INSTALL_PREFIX="$3" -DBUILD_GMOCK=OFF -DINSTALL_GTEST=ON \
                     && cmake --build "$2" -j "$4" --target install' _ \
                "$GTEST_SRC" "$PROJECT_DIR/build/googletest-build" "$GTEST_INSTALL" "$JOBS"
        write_stamp "$GTEST_INSTALL" "$GTEST_STAMP"
        GTEST_ROOT_DIR="$GTEST_INSTALL"
    fi
fi

#==============================================================================
# RandBLAS itself.
#==============================================================================
RANDBLAS_BUILD="$PROJECT_DIR/build/RandBLAS-build"
mkdir -p "$RANDBLAS_BUILD"

RB_ARGS=(
    -S "$REPO_DIR" -B "$RANDBLAS_BUILD"
    -DCMAKE_BUILD_TYPE=Release
    -Dblaspp_DIR="$BLASPP_CMAKE_DIR"
    -DRandom123_DIR="$RANDOM123_DIR"
    -DCMAKE_INSTALL_PREFIX="$RANDBLAS_INSTALL_DIR"
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
)
if (( WANT_TESTS )); then
    RB_ARGS+=(-DBUILD_TESTS=ON)
    if [[ -n "$GTEST_ROOT_DIR" ]]; then
        RB_ARGS+=(-DGTest_ROOT="$GTEST_ROOT_DIR")
    fi
else
    RB_ARGS+=(-DBUILD_TESTS=OFF)
fi
RB_ARGS+=("${OPENMP_FLAGS[@]}")

run_step "Configuring RandBLAS" cmake "${RB_ARGS[@]}"
run_step "Building and installing RandBLAS" \
    cmake --build "$RANDBLAS_BUILD" -j "$JOBS" --target install

#==============================================================================
# Verification.
#
# Compile, link and *run* a program against the freshly installed RandBLAS.
# Configuring successfully is not the same as producing something that works:
# this is what catches a BLAS that resolves at configure time but fails to
# link, a missing runtime library, and a width that is not what was asked for.
# It tests through BLAS++ rather than against raw dgemm_ because that is how
# RandBLAS actually reaches the BLAS.
#==============================================================================
CONFTEST_DIR="$PROJECT_DIR/build/conftest"
rm -rf "$CONFTEST_DIR"
mkdir -p "$CONFTEST_DIR/src"

cat > "$CONFTEST_DIR/src/CMakeLists.txt" <<'CONFTEST_CMAKE'
cmake_minimum_required(VERSION 3.21)
project(randblas_conftest CXX)
find_package(RandBLAS REQUIRED)
add_executable(conftest conftest.cc)
target_link_libraries(conftest RandBLAS)
CONFTEST_CMAKE

cat > "$CONFTEST_DIR/src/conftest.cc" <<'CONFTEST_CC'
// Sketch a small matrix and multiply through BLAS++, then check the numbers.
// A wrong-width or half-linked BLAS shows up here rather than in a user's
// first real run.
#include <RandBLAS.hh>
#include <blas.hh>
#include <blas/defines.h>
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    // BLAS++ bakes BLAS_ILP64 into its installed blas/defines.h, so this
    // reports the width the headers were configured for. The blas_int typedef
    // itself is not reachable from <blas.hh>, and sizeof() on it would report
    // the same header-level fact anyway -- the gemm check below is what
    // actually exercises the linked library.
#if defined(BLAS_ILP64)
    std::printf("blas_ilp64=1\n");
#else
    std::printf("blas_ilp64=0\n");
#endif

    const int64_t m = 8, n = 4;
    std::vector<double> S(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState state(0);
    RandBLAS::fill_dense(D, S.data(), state);

    // C = S^T S, which must be symmetric positive semidefinite.
    std::vector<double> C(n * n, 0.0);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               n, n, m, 1.0, S.data(), m, S.data(), m, 0.0, C.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        if (!(C[i + i * n] > 0.0) || !std::isfinite(C[i + i * n])) {
            std::printf("FAIL: diagonal entry %lld is %f\n", (long long)i, C[i + i * n]);
            return 1;
        }
        for (int64_t j = 0; j < n; ++j) {
            if (std::fabs(C[i + j * n] - C[j + i * n]) > 1e-12) {
                std::printf("FAIL: not symmetric at (%lld,%lld)\n", (long long)i, (long long)j);
                return 1;
            }
        }
    }
    std::printf("OK\n");
    return 0;
}
CONFTEST_CC

verify_install() {
    cmake -S "$CONFTEST_DIR/src" -B "$CONFTEST_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$RANDBLAS_INSTALL_DIR" \
        -Dblaspp_DIR="$BLASPP_CMAKE_DIR" \
        -DRandom123_DIR="$RANDOM123_DIR" \
        "${OPENMP_FLAGS[@]}" >> "$LOG" 2>&1
    cmake --build "$CONFTEST_DIR/build" -j "$JOBS" >> "$LOG" 2>&1
    "$CONFTEST_DIR/build/conftest" > "$CONFTEST_DIR/output.txt" 2>&1
    grep -q '^OK$' "$CONFTEST_DIR/output.txt"
}
run_step "Verifying the install links and runs" verify_install
cat "$CONFTEST_DIR/output.txt" >> "$LOG"

# Read the width back from what was actually compiled, rather than trusting
# the value this script asked for. They differ whenever BLAS++ came from
# somewhere else -- BLASPP_INSTALL_DIR, a system package, a previous run --
# and that is exactly the case worth reporting accurately.
case "$(sed -n 's/^blas_ilp64=//p' "$CONFTEST_DIR/output.txt" | head -n1)" in
    1) OBSERVED_WIDTH="ILP64 (64-bit BLAS integers)" ;;
    0) OBSERVED_WIDTH="LP64 (32-bit BLAS integers)" ;;
    *) OBSERVED_WIDTH="unknown" ;;
esac

#==============================================================================
# Examples.
#
# Not part of the core install, and not merely because they take time: they
# need two dependencies RandBLAS itself does not (LAPACK++ and
# fast_matrix_market), and examples/CMakeLists.txt requires OpenMP outright,
# which stock Apple Clang cannot provide. Making them a separate opt-in keeps
# a plain install from failing over something the library does not need.
#==============================================================================
build_examples() {
    if [[ -n "${LAPACKPP_INSTALL_DIR:-}" ]] && \
       [[ -n "$(find_cmake_config "$LAPACKPP_INSTALL_DIR" lapackpp)" ]]; then
        LAPACKPP_CMAKE_DIR="$(find_cmake_config "$LAPACKPP_INSTALL_DIR" lapackpp)"
        skip_step "LAPACK++ source ... using $LAPACKPP_INSTALL_DIR"
        skip_step "LAPACK++ ... reusing external install"
    else
        local src="$PROJECT_DIR/lib/lapackpp"
        local stamp="$LAPACKPP_URL@$LAPACKPP_REF blaspp=$BLASPP_CMAKE_DIR"
        if (( ! FRESH )) && stamp_matches "$LAPACKPP_INSTALL" "$stamp"; then
            skip_step "LAPACK++ source ... already present"
            skip_step "LAPACK++ ... reusing existing install"
        else
            run_step "Fetching LAPACK++ ($LAPACKPP_REF)" \
                clone_pinned "$LAPACKPP_URL" "$src" "$LAPACKPP_REF"
            run_step "Building and installing LAPACK++" \
                bash -c 'cmake -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release \
                            -DCMAKE_INSTALL_PREFIX="$3" -Dblaspp_DIR="$4" \
                            -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
                         && cmake --build "$2" -j "$5" --target install' _ \
                    "$src" "$PROJECT_DIR/build/lapackpp-build" "$LAPACKPP_INSTALL" \
                    "$BLASPP_CMAKE_DIR" "$JOBS"
            write_stamp "$LAPACKPP_INSTALL" "$stamp"
        fi
        LAPACKPP_CMAKE_DIR="$(find_cmake_config "$LAPACKPP_INSTALL" lapackpp)"
    fi

    run_step "Configuring examples" \
        cmake -S "$REPO_DIR/examples" -B "$PROJECT_DIR/build/examples-build" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_PREFIX_PATH="$RANDBLAS_INSTALL_DIR" \
            -Dblaspp_DIR="$BLASPP_CMAKE_DIR" \
            -Dlapackpp_DIR="$LAPACKPP_CMAKE_DIR" \
            -DRandom123_DIR="$RANDOM123_DIR" \
            -DFETCHCONTENT_BASE_DIR="$PROJECT_DIR/build/fetchcontent-cache" \
            "${OPENMP_FLAGS[@]}"
    run_step "Building examples" \
        cmake --build "$PROJECT_DIR/build/examples-build" -j "$JOBS"
}

# Reproduce this invocation with --examples added, for the offer below. Every
# option that changed the result has to be carried across, or the printed
# command silently builds something different from what was just installed --
# and for --blas=custom it would not run at all.
#
# Written as if-blocks rather than "[[ test ]] && append": under set -e a
# false test at the end of a list exits the script.
EXAMPLES_COMMAND="bash $SCRIPT_DIR/install.sh --examples --blas=$BLAS_BACKEND --project-dir=$PROJECT_DIR"
if [[ -n "$PREFIX_OVERRIDE" ]]; then
    EXAMPLES_COMMAND+=" --prefix=$PREFIX_OVERRIDE"
fi
if [[ -n "$BLAS_LIBRARIES_ARG" ]]; then
    EXAMPLES_COMMAND+=" --blas-libraries='$BLAS_LIBRARIES_ARG'"
fi
if [[ "$BLAS_INT_CHOICE" != "auto" ]]; then
    EXAMPLES_COMMAND+=" --blas-int=$BLAS_INT_CHOICE"
fi
if (( ! WANT_TESTS  )); then EXAMPLES_COMMAND+=" --no-tests"; fi
if (( ! WANT_OPENMP )); then EXAMPLES_COMMAND+=" --no-openmp"; fi

if (( WANT_EXAMPLES )); then
    build_examples
fi

#==============================================================================
# Summary.
#==============================================================================
printf '\n%s%sRandBLAS installed successfully.%s\n\n' "$C_OK" "$C_BOLD" "$C_OFF"
printf '  Backend            %s, %s\n' "$BLAS_BACKEND" "$OBSERVED_WIDTH"
printf '  OpenMP             %s\n' "$( ((WANT_OPENMP)) && echo enabled || echo disabled )"
printf '  Project layout     %s\n' "$PROJECT_DIR"
printf '  Installed library  %s\n' "$RANDBLAS_INSTALL_DIR"
if (( WANT_EXAMPLES )); then
    printf '  Examples           %s\n' "$PROJECT_DIR/build/examples-build"
fi
printf '  Full build log     %s\n' "$LOG"

if (( ${#WARNINGS[@]} )); then
    printf '\n%s%d warning(s) from this run:%s\n' "$C_WARN" "${#WARNINGS[@]}" "$C_OFF"
    for w in "${WARNINGS[@]}"; do
        printf '  - %s\n' "$w"
    done
fi

if (( WANT_TESTS )); then
    printf '\n  Run the test suite:\n    ctest --test-dir %s\n' "$RANDBLAS_BUILD"
fi

printf '\n  Consume from CMake with:\n    -DRandBLAS_DIR=%s\n' \
    "$(find_cmake_config "$RANDBLAS_INSTALL_DIR" RandBLAS)"

# The examples offer. Interactive users get asked; everyone else gets the
# command, so the option is discoverable either way rather than living only in
# --help where nobody looks after a successful install.
if (( ! WANT_EXAMPLES )); then
    printf '\n  The examples are not built by default: they additionally need\n'
    printf '  LAPACK++ and fast_matrix_market, and they require OpenMP.\n'
    if (( INTERACTIVE )) && ask "  Build them now?" n; then
        TOTAL_STEPS=$(( STEP + 4 ))
        printf '\n'
        build_examples
        printf '\n%s%sExamples built.%s\n    %s\n' "$C_OK" "$C_BOLD" "$C_OFF" \
            "$PROJECT_DIR/build/examples-build"
    else
        printf '  To build them later, re-run with --examples:\n    %s\n' "$EXAMPLES_COMMAND"
    fi
fi

printf '\n'
