# OpenCL on Apple M1 under Fedora Asahi — driver work and measurement

Making Mesa's OpenCL implementation (Rusticl, on the Gallium Asahi driver)
faster and more capable on M1-family Macs running Fedora Asahi Remix, and
recording how each change was measured and checked.

The driver changes are commits on branch `opencl` of
[aquarat/mesa](https://github.com/aquarat/mesa), stacked on the Honeykrisp
(Vulkan) work from [got-bringup](https://github.com/aquarat/got-bringup).
`mesa-source.env` pins the exact commit. This repository holds the build
scripts, benchmarks, correctness tests, conformance-test harness and the
written record. **Start with [STATE.md](STATE.md).** Nothing here is
upstream Mesa; anyone is welcome to take it there.

## What changed (Mac mini M1, 8-core GPU)

| | before | after |
|---|---:|---:|
| chain of small kernels that each write a buffer | 38 us/kernel | 4.2 us/kernel |
| 64 independent one-workgroup kernels | 37 ms | 1.6 ms |
| reading a GPU-written buffer back to the CPU | 0.24 GB/s | 9.3 GB/s |
| reading back a 371 MB image | 1.6 s | 76 ms |
| counted loop with a small body | 795 GFLOPS | ~2,100 GFLOPS |
| darktable 5.6.1 export of a 24 MP raw | could not use the GPU | 0.61-0.65 s (CPU: 1.41 s) |

The last row is a compatibility fix: the build embeds a patched libclc,
without which precise `sin`, `cos`, `pow`, `atan2` and `hypot` fail to link
under Fedora's libclc. Details, method, trade-offs and what did not work
are in STATE.md.

## Build and run

Needs Docker and an M1-family Mac on Fedora Asahi Remix 44. Nothing is
installed system-wide.

    ./build-in-container.sh fetch    # clone the pinned Mesa commit into ./mesa
    ./build-in-container.sh setup    # Fedora 44 container, deps, mesa-libclc, meson
    ./build-in-container.sh build    # build into ./install
    ./with-cl.sh clinfo              # any OpenCL program, on the local build

`with-cl.sh` points the ICD loader at `./install` for one process and gives
it its own shader cache; the system Rusticl stays the default everywhere
else.

**Run GPU tests memory-capped.** GPU buffer objects on Asahi are pinned
memory the kernel's OOM killer cannot attribute to the process that owns
them, so a test that over-allocates gets unrelated processes killed instead
(this happened during this work). They are charged to the process's cgroup,
so a cap contains them:

    ./capped.sh 10G build-logs/cts.log ./cts-run.sh mylabel local

## Switches for A/B testing

Every performance change can be turned off at run time:

| variable | restores |
|---|---|
| `ASAHI_PERFTEST=barrierflush` | a GPU submission after every buffer-writing kernel |
| `ASAHI_PERFTEST=nooverlap` | a full GPU barrier after every dispatch |
| `ASAHI_PERFTEST=cpuread` | CPU reads of images straight from uncached memory |
| `RUSTICL_DEBUG=wc_buffers` | uncached (write-combined) allocation for all buffers |
| `AGX_MESA_DEBUG=nounroll` | no partial loop unrolling |

`AGX_CDM_BARRIER_MASK` overrides the weak barrier's cache bits (default
`0x80`), `AGX_BO_CACHE_MB` the GPU buffer cache cap (default RAM/8, at most
2 GiB). Any `AGX_MESA_DEBUG` flag also disables the shader disk cache, so
when comparing with and without one, disable the cache on both sides
(`MESA_SHADER_CACHE_DISABLE=1`) or the flagged side pays compile time.

## Layout

| path | what |
|---|---|
| `STATE.md` | findings, numbers, method; start here |
| `mesa-source.env` | Mesa fork/branch/commit and mesa-libclc commit |
| `build-in-container.sh`, `build-libclc.sh` | build the driver in a container |
| `with-cl.sh` | run one program on the local build |
| `capped.sh` | run a command in a memory-capped systemd scope |
| `bench/` | microbenchmarks, one question each (header comment says which) |
| `tests/hazard.c` | ordering test for dispatch overlap |
| `tests/loops.c` | correctness test for the loop-unroll pass |
| `cts-run.sh`, `cts-compare.sh` | run Khronos OpenCL CTS suites, compare failure sets |
| `scrub-logs.sh` | strip host name and home paths from result logs |
| `results/`, `cts-results/` | raw outputs behind the numbers |
| `workloads/darktable/` | the darktable export used as a real workload |

`make` builds `bench/` and `tests/` against the system ICD loader. The CTS
itself is not vendored: clone
[KhronosGroup/OpenCL-CTS](https://github.com/KhronosGroup/OpenCL-CTS) into
`cts-src/` and build it into `cts-build/` (it needs SPIR-V headers and
`spirv-as` from spirv-tools).

## License

MIT (see LICENSE), for the scripts, tests and benchmarks here. The Mesa
changes are under Mesa's own licenses.
