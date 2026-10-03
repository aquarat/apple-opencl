#!/bin/bash
# Run a command against the privately built Rusticl instead of the system one.
#   ./with-cl.sh clinfo
# The system ICD (/etc/OpenCL/vendors) is hidden for this process only.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREFIX=${PREFIX:-$HERE/install}
ICD_DIR=$HERE/.icd
mkdir -p "$ICD_DIR"
echo "$PREFIX/lib64/libRusticlOpenCL.so.1" > "$ICD_DIR/rusticl-local.icd"
export OCL_ICD_VENDORS=$ICD_DIR
# A separate shader cache, so a codegen change can never be served stale
# binaries built by the system driver (got-bringup measurement-hazards.md).
export MESA_SHADER_CACHE_DIR=${MESA_SHADER_CACHE_DIR:-$HOME/.cache/mesa-opencl-local}
exec "$@"
