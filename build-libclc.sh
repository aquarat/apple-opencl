#!/bin/bash
# Build mesa-libclc (Karol Herbst's pinned libclc fork) for the mesa3d SPIR-V
# targets inside the build container, into ./libclc-install, where
# build-in-container.sh picks it up (embedded via -Dstatic-libclc=all).
# Repository, branch and commit come from mesa-source.env.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=mesa-source.env
. "$HERE/mesa-source.env"
NAME=${CONTAINER:-rusticl-build}
SRC=$HERE/mesa-libclc
if [ ! -d "$SRC/.git" ]; then
  git clone -b "$LIBCLC_BRANCH" "$LIBCLC_REPO" "$SRC" || exit 1
fi
git -C "$SRC" cat-file -e "$LIBCLC_COMMIT^{commit}" 2>/dev/null || git -C "$SRC" fetch origin "$LIBCLC_BRANCH"
git -C "$SRC" checkout -q --detach "$LIBCLC_COMMIT" || { echo "checkout $LIBCLC_COMMIT failed"; exit 1; }
echo "mesa-libclc: $(git -C "$SRC" log -1 --format='%h %s')"
docker exec "$NAME" bash -c "dnf -y -q install --setopt=install_weak_deps=False cmake clang llvm spirv-llvm-translator-tools zstd >/dev/null &&
  cd '$SRC' && rm -rf build &&
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX='$HERE/libclc-install' \
    -DLIBCLC_TARGETS_TO_BUILD='spirv-mesa3d-;spirv64-mesa3d-' >/dev/null &&
  nice -n 10 ninja -C build -j3 >/dev/null && ninja -C build install >/dev/null" || { echo "libclc build failed"; exit 1; }
ls -la "$HERE"/libclc-install/share/mesa-clc/
