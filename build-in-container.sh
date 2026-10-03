#!/bin/bash
# Build Rusticl (Mesa's OpenCL) on the Gallium Asahi driver, from the Mesa fork
# and commit pinned in mesa-source.env, inside a Fedora 44 container, into a
# private prefix. Nothing is installed on the host.
#
# Usage: ./build-in-container.sh <command>
#   fetch      clone the Mesa fork into $MESA_SRC and check out the pinned commit
#   setup      create the container, install build deps, build mesa-libclc
#              (build-libclc.sh), meson setup
#   configure  meson setup only (for a second BUILDDIR/PREFIX)
#   build      ninja + install into $PREFIX
#   all        fetch + setup + build
#   shell      a shell in the container
#
# Environment:
#   MESA_SRC   Mesa source tree (default ./mesa). May be a git worktree; its
#              common git directory is mounted into the container as well.
#   PREFIX     install prefix (default ./install)
#   BUILDDIR   meson build directory inside MESA_SRC (default build). A second
#              tree (BUILDDIR=build-dev PREFIX=$PWD/install-dev) lets
#              experiments build while a CTS run is still loading ./install.
#   JOBS       compile jobs and container CPUs (default 3)
#
# Run anything against the result with ./with-cl.sh <command>.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=mesa-source.env
. "$HERE/mesa-source.env"
NAME=${CONTAINER:-rusticl-build}
SRC=${MESA_SRC:-$HERE/mesa}
PREFIX=${PREFIX:-$HERE/install}
BUILDDIR=${BUILDDIR:-build}
LOG=$HERE/build-logs
JOBS=${JOBS:-3}
mkdir -p "$LOG"

MESON_ARGS=(--prefix="$PREFIX" --libdir=lib64 --buildtype=release
  -Dgallium-drivers=asahi -Dgallium-rusticl=true -Dgallium-rusticl-enable-drivers=asahi
  -Dvulkan-drivers= -Dplatforms= -Dopengl=false -Dgles1=disabled -Dgles2=disabled
  -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dglvnd=disabled -Dvideo-codecs=
  -Dtools= -Dvulkan-layers= -Dbuild-tests=false -Dllvm=enabled -Dshared-llvm=enabled
  -Dxmlconfig=enabled -Dshader-cache=enabled -Dstatic-libclc=all)
# The patched libclc (mesa-libclc, llvm_22 branch), built by build-libclc.sh and
# embedded: Fedora's upstream libclc lacks precise sin/cos/pow/atan2/hypot for
# the mesa3d SPIR-V target, so programs using them without
# -cl-fast-relaxed-math fail to build ("nir_shader not fully linked").
PCPATH=$HERE/libclc-install/share/pkgconfig

die() { echo "build-in-container: $*" >&2; exit 1; }

fetch() {
  if [ ! -e "$SRC/.git" ]; then
    git clone --branch "$MESA_BRANCH" "$MESA_REPO" "$SRC" || die "clone failed"
  fi
  git -C "$SRC" cat-file -e "$MESA_COMMIT^{commit}" 2>/dev/null ||
    git -C "$SRC" fetch origin "$MESA_BRANCH" || die "fetch failed"
  local head
  head=$(git -C "$SRC" rev-parse HEAD)
  if [ "${head:0:12}" != "${MESA_COMMIT:0:12}" ]; then
    if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
      echo "warning: $SRC has local changes and is not at the pinned $MESA_COMMIT; leaving it alone"
    else
      git -C "$SRC" checkout -q --detach "$MESA_COMMIT" || die "checkout $MESA_COMMIT failed"
    fi
  fi
  echo "Mesa at $(git -C "$SRC" log -1 --format='%h %s')"
}

meson_setup() {
  [ -f "$PCPATH/mesa-libclc.pc" ] || die "no mesa-libclc in ./libclc-install: run ./build-libclc.sh"
  docker exec "$NAME" bash -c "cd '$SRC' && rm -rf '$BUILDDIR' && PKG_CONFIG_PATH='$PCPATH' nice -n 10 meson setup '$BUILDDIR' ${MESON_ARGS[*]}" \
    > "$LOG/meson-$BUILDDIR.log" 2>&1 || { tail -30 "$LOG/meson-$BUILDDIR.log"; die "meson failed"; }
  echo "configured $BUILDDIR -> $PREFIX"
}

setup() {
  [ -e "$SRC/.git" ] || die "no Mesa tree at $SRC: run '$0 fetch' first"
  # A worktree keeps its objects in the main repository's git directory.
  local gitcommon mounts=(-v "$SRC:$SRC" -v "$HERE:$HERE")
  gitcommon=$(git -C "$SRC" rev-parse --path-format=absolute --git-common-dir)
  case "$gitcommon" in "$SRC"/*) ;; *) mounts+=(-v "$gitcommon:$gitcommon") ;; esac

  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --memory 5g --memory-swap 7g --cpus "$JOBS" \
    "${mounts[@]}" -e HOME=/root fedora:44 sleep infinity >/dev/null || die "docker run failed"
  docker exec "$NAME" bash -c 'dnf -y install --setopt=install_weak_deps=False gcc gcc-c++ meson ninja-build bison flex python3-devel python3-mako python3-ply python3-pyyaml python3-packaging python3-pycparser libdrm-devel expat-devel zlib-devel libzstd-devel llvm-devel clang-devel spirv-tools-devel spirv-llvm-translator-devel libclc-devel git-core glslang rust cargo bindgen-cli rustfmt' \
    > "$LOG/dnf.log" 2>&1 || { tail -20 "$LOG/dnf.log"; die "dnf failed"; }
  docker exec "$NAME" git config --global --add safe.directory '*'
  "$HERE/build-libclc.sh" || die "mesa-libclc build failed"
  meson_setup
  echo "setup ok"
}

build() {
  [ -d "$SRC/$BUILDDIR" ] || die "no build directory $SRC/$BUILDDIR: run '$0 setup' (or set MESA_SRC)"
  local L
  L="$LOG/ninja-$(date +%Y%m%d-%H%M%S).log"
  docker exec "$NAME" bash -c "cd '$SRC' && nice -n 10 ninja -C '$BUILDDIR' -j$JOBS && ninja -C '$BUILDDIR' install >/dev/null" > "$L" 2>&1
  local rc=$?
  echo "ninja rc=$rc ($L)"
  [ $rc -ne 0 ] && grep -n "error\|FAILED" "$L" | tail -20
  ls -la "$PREFIX"/lib64/libRusticlOpenCL.so* 2>/dev/null
  return $rc
}

case "${1:-}" in
fetch)     fetch ;;
setup)     setup ;;
configure) meson_setup ;;
build)     build ;;
all)       fetch && setup && build ;;
shell)     docker exec -it "$NAME" bash ;;
*)         sed -n '2,24p' "$0"; exit 2 ;;
esac
