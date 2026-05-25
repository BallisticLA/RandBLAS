#!/usr/bin/env bash
# Build the TSan dev image if missing, then drop into a shell with the repo
# mounted at /work/RandBLAS. Subcommands:
#
#   ./docker/tsan/run.sh                  # interactive shell in the container
#   ./docker/tsan/run.sh build-and-test   # configure, build, and ctest under TSan
#   ./docker/tsan/run.sh rebuild-image    # force-rebuild the dev image
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image_tag="randblas-tsan"
subcommand="${1:-shell}"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found on PATH" >&2
    exit 1
fi

ensure_image() {
    if ! docker image inspect "${image_tag}" >/dev/null 2>&1; then
        docker build -t "${image_tag}" "${repo_root}/docker/tsan/"
    fi
}

case "${subcommand}" in
    rebuild-image)
        docker build --no-cache -t "${image_tag}" "${repo_root}/docker/tsan/"
        ;;

    shell)
        ensure_image
        docker run --rm -it \
            -v "${repo_root}:/work/RandBLAS" \
            "${image_tag}"
        ;;

    build-and-test)
        ensure_image
        # Mount the source read-only so the in-container build dir can't pollute
        # the host tree; build artifacts live in a separate volume so they
        # survive between runs and don't clash with the macOS host build.
        docker run --rm \
            -v "${repo_root}:/work/RandBLAS:ro" \
            -v "randblas-tsan-build:/work/build" \
            "${image_tag}" \
            bash -c '
                set -euxo pipefail
                cd /work/build
                cmake \
                    -DCMAKE_C_COMPILER=clang \
                    -DCMAKE_CXX_COMPILER=clang++ \
                    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                    -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer" \
                    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
                    -Dblaspp_DIR="${blaspp_DIR}" \
                    -DRandom123_DIR="${Random123_DIR}" \
                    /work/RandBLAS
                make -j"$(nproc)"
                # ignore_noninstrumented_modules=1 silences spurious reports
                # against blaspp/openblas/libomp/libc, none of which are built
                # with TSan. Without it, the run drowns in false positives.
                OMP_NUM_THREADS=4 \
                TSAN_OPTIONS="halt_on_error=0 ignore_noninstrumented_modules=1 second_deadlock_stack=1" \
                    ctest --output-on-failure -E test_rng_speed
            '
        ;;

    *)
        echo "usage: $0 [shell | build-and-test | rebuild-image]" >&2
        exit 1
        ;;
esac
