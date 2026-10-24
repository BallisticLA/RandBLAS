# Dev container for running ThreadSanitizer locally on macOS Apple Silicon,
# where TSan on the host is broken (shadow-memory mapping fails on arm64
# Darwin). The container is Linux, where TSan + libomp is reliable on both
# arm64 and amd64.
#
# Build:   ./docker/tsan/run.sh rebuild-image   (run.sh builds on first use)
# Run:     ./docker/tsan/run.sh
#
# The image bakes blaspp + Random123 into /opt so that iterating on RandBLAS
# itself is fast — only RandBLAS gets rebuilt per session. BLAS++ comes from
# installers/dependency-pins.env, which run.sh passes in as build arguments
# and also folds into the image tag, so changing the pin builds a new image.
# A bare `docker build` without them fails rather than guessing a commit.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
        ca-certificates git build-essential gfortran cmake make \
        clang libomp-dev libclang-rt-18-dev llvm-18 \
        libopenblas-openmp-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/llvm-symbolizer-18 /usr/local/bin/llvm-symbolizer
# ^ libclang-rt-18-dev ships the compiler-rt sanitizer runtime archives
#   (incl. libclang_rt.tsan-*.a); without it clang -fsanitize=thread fails to
#   link. llvm-18 + the llvm-symbolizer symlink give TSan readable file:line
#   stack traces instead of raw hex addresses.

ENV CC=clang
ENV CXX=clang++

# blaspp: fetched by commit hash, which must come back as itself. Declared
# here rather than at the top so that a new pin does not invalidate the
# apt layer above.
ARG BLASPP_URL
ARG BLASPP_REF
RUN test -n "$BLASPP_URL" && test -n "$BLASPP_REF" \
        || { echo "BLASPP_URL and BLASPP_REF are required build arguments; use docker/tsan/run.sh" >&2; exit 1; } \
    && git init --quiet /tmp/blaspp \
    && git -C /tmp/blaspp fetch --quiet --depth 1 "$BLASPP_URL" "$BLASPP_REF" \
    && git -C /tmp/blaspp checkout --quiet --detach FETCH_HEAD \
    && test "$(git -C /tmp/blaspp rev-parse HEAD)" = "$BLASPP_REF" \
    && cmake -S /tmp/blaspp -B /tmp/blaspp-build \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/blaspp-install \
        -Dblas=openblas \
        -Dbuild_tests=OFF \
    && cmake --build /tmp/blaspp-build -j"$(nproc)" --target install \
    && rm -rf /tmp/blaspp /tmp/blaspp-build

# Random123: header-only. cp -R is portable (BSD/GNU) so we don't need the
# Random123 Makefile's `install-include` target (which uses GNU `cp -d`).
RUN git clone --depth 1 https://github.com/DEShawResearch/Random123.git /tmp/Random123 \
    && mkdir -p /opt/Random123-install/include \
    && cp -R /tmp/Random123/include/Random123 /opt/Random123-install/include/ \
    && rm -rf /tmp/Random123

ENV blaspp_DIR=/opt/blaspp-install/lib/cmake/blaspp
ENV Random123_DIR=/opt/Random123-install/include

WORKDIR /work
CMD ["/bin/bash"]
