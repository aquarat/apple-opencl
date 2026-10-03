# OpenCL on the M1 (Rusticl on Gallium Asahi) — where this stands

Machine for every number here unless stated: Mac mini M1 (T8103, GPU G13G B1,
8 cores, 16 GB), Fedora Asahi Remix 44, kernel 7.1.13-402.asahi, system Mesa
26.1.8. Driver under test: branch `opencl` of
[aquarat/mesa](https://github.com/aquarat/mesa) (`mesa-source.env`), built by
`build-in-container.sh`, run with `with-cl.sh`. Work done 2026-10-02/03.

## The changes

| Mesa commit | area | change |
|---|---|---|
| `9e79757` | asahi (Gallium) | no submission per buffer-writing kernel; independent dispatches overlap |
| `44caf37` | Rusticl | buffers the host may read back are allocated cached (write-back) |
| `c2e4e40` | asahi (Gallium) | large reads of uncached textures go through a GPU staging blit |
| `68451a0` | NIR + AGX compiler | partial unrolling of counted loops, with a remainder loop |
| `817bbf4` | asahi (Gallium + lib) | fixes a leak `c2e4e40` exposed, caps the BO cache, makes staged reads bit-exact |
| `51e56ca` | asahi (lib) | BO cache cap sized at RAM/8 (RAM/16 cost darktable a third of its speed) |
| (build) | mesa-libclc embedded | precise math built-ins that Fedora's libclc lacks |

Each change has a run-time off switch (README). Scope note: the loop-unroll
pass runs in the AGX compiler for every compute shader, and the BO cache cap
is in the library Honeykrisp shares, so a Honeykrisp (Vulkan) built from this
branch also gets those two. Only OpenCL has been tested here.

## Baseline (system Mesa 26.1.8)

Rusticl is on by default for asahi and reports OpenCL 3.0 / FULL_PROFILE with
a passed conformance run (v2024-08-08). clpeak `--opencl` (`results/`):

| | |
|---|---|
| fp32 / fp16 | ~2.25-2.55 TFLOPS (peak 8 x 128 x 2 x 1.278 GHz = 2.62) |
| int32 | ~0.65 TOPS (1/4 rate) |
| global / local memory | 56-64 GB/s / 0.9-1.2 TB/s |
| kernel launch round trip | 250 us (Vulkan backend: 218 us) |

clpeak's scalar mixed-precision figure (549 GFLOPS) is a single dependent
chain per work-item, i.e. latency-bound by design; its mp8/mp16 figures from
clpeak 2.1.1 (55 TFLOPS) are a benchmark artefact. Current clpeak mp kernels
produce results identical to a NumPy reference.

## 1. One GPU submission per kernel, and no overlap (`9e79757`)

`agx_memory_barrier()` flushed any batch with "incoherent writes", and
Rusticl issues a buffer barrier after every launch, so every OpenCL kernel
that wrote a buffer became its own GPU submission. On top of that every
compute dispatch was followed by a full CDM barrier, so independent kernels
never overlapped -- the same finding as got-bringup's Honeykrisp work.

Within a compute batch the barrier after each dispatch already orders
buffer writes; hazards against other batches are tracked per resource; the
host only reads after a fence. So buffer-only barriers no longer flush a
compute batch (image barriers still do). Dispatches from `launch_grid` now
get a weak barrier (bit 7, as Honeykrisp), with the full one owed and
settled before any dispatch that touches a buffer an overlapped dispatch
wrote (or writes one it read; global buffers count as written), before
every driver-internal, indirect or blitter dispatch, when the 64-entry
tracking table fills, and at the end of the stream. Shared compute scratch
counts as a written buffer.

| `bench/overlap.c` | system 26.1.8 | old behaviour* | barrier fix | + overlap |
|---|---:|---:|---:|---:|
| dependent tiny writing kernels, per kernel | 37.4 us | 37.7 us | 4.4 us | **4.2 us** |
| 64 independent 1-workgroup dispatches | 39.0 ms | 38.8 ms | 36.7 ms | **1.6 ms** |
| same work as 1 dispatch x 64 groups | 0.88 ms | 0.83 ms | 0.79 ms | 0.75 ms |

\* same build, `ASAHI_PERFTEST=barrierflush,nooverlap`.

`tests/hazard.c` (RAW/WAR/WAW across overlapping dispatches with a slow
producer, sub-buffers, concurrent scratch users, tracking-table overflow,
fill/copy helpers after a slow producer) passes 30/30 repetitions.

## 2. Host reads of GPU results ran at 0.25 GB/s (`44caf37`, `c2e4e40`, `817bbf4`)

Asahi allocates everything write-combined unless it is a staging resource,
and CPU reads of write-combined memory are uncached loads at DRAM latency.
GPU-side bandwidth does not depend on the CPU cache mode (59.9 GB/s copy
either way, `bench/wbbench.c`), so this was pure loss on the host side.

**Buffers** (`44caf37`, Rusticl): buffers the host may read back are
allocated write-back. Kernel-`READ_ONLY`, `HOST_WRITE_ONLY` and
`HOST_NO_ACCESS` buffers stay write-combined, because host writes to those
are twice as fast. `bench/hostxfer.c`, 64 MiB READ_WRITE buffer:

| | before | after |
|---|---:|---:|
| ReadBuffer | 0.24 GB/s | 9.3 GB/s |
| map for reading | 0.24 GB/s | 3.6 GB/s |
| WriteBuffer (the price) | 24.6 GB/s | 11.3 GB/s |

**Images** (`c2e4e40`, fixed in `817bbf4`): CPU reads of at least 64 KiB of
a write-combined 2D texture go through a GPU blit into a linear write-back
staging texture. `bench/imgread.c`, 6048x4024 RGBA32F, every pixel and odd
sub-regions verified: ReadImage **1621 ms -> 76 ms**.

`c2e4e40` alone measured 28 ms but was wrong in two ways, both found by the
CTS:

* **It exposed a leak that took the machine down.** `asahi_compute_restore`
  handed the saved sampler view back to `set_sampler_views`, which takes its
  own reference, and then merely cleared the pointer, so every compute blit
  made while a compute sampler view was bound leaked that view and its
  texture. Under OpenCL that is every staged readback after a kernel that
  read an image: ~0.77 GB per pass of the CTS `imagedim` sizes
  (`bench/imgleak.c`). The leaked memory is pinned GPU memory the OOM
  killer cannot attribute, so it killed unrelated processes, twice. The bug
  predates this work; compute blits were simply rare under Rusticl before.
  Fixed, along with two contributing problems: the BO cache had no size cap
  (now RAM/8, at most 2 GiB, with single BOs cached up to half of that), and
  the blit's own image and view stayed bound after it, keeping a staging
  texture the size of the transfer alive. A first cap of RAM/16 with BOs
  over a quarter of it never cached was too tight: darktable re-allocates
  371 MiB buffers, and returning them to the kernel every time took its
  export from 0.63 to 0.92 s (`51e56ca`).
* **It was not bit-exact.** The blit went through the format's own view,
  i.e. through float: SNORM does not round-trip and Mesa's SNORM-to-SINT
  helper misses the swizzled layouts (CTS `basic imagearraycopy`, ARGB/ABGR
  SNORM_INT8), and this GPU flushes FP32 denormals. Staged reads now copy
  texels as a raw unsigned integer format of the same size (tiling depends
  only on texel size): exact for every format, and the reason for 76 ms
  rather than 28.

The path is limited to 2D and rectangle textures; arrays, 3D and cubes keep
the CPU path until validated. CTS `basic`, all 20 image tests: pass in
135 s against 402 s on the CPU path; peak GPU memory 5.0 GB at `51e56ca`
(3.1 GB on the CPU path; the difference is BO cache, bounded at 2 GiB).

## 3. darktable could not use the GPU at all (mesa-libclc)

Fedora's libclc lacks precise `sin`, `cos`, `pow`, `atan2`, `hypot` (and
likely more) for the `spirv64-mesa3d-` target. Any program using them
without `-cl-fast-relaxed-math` fails to build with "nir_shader not fully
linked" -- what Rusticl's "Patched Mesa libclc not detected" warning is
about. darktable 5.6.1 compiles without that flag: 7 of its 42 kernel files
failed, including `basic.cl`, so it rejected the device.

The build embeds mesa-libclc (Karol Herbst's pinned fork, branch `llvm_22`,
`build-libclc.sh`) with `-Dstatic-libclc=all`, so the driver no longer
depends on Fedora's libclc at run time. All 42 darktable kernel files
compile and the warning is gone.

darktable-cli default export of a 24 MP Sony ARW, `workloads/darktable/`:

| | pipeline |
|---|---:|
| CPU (8 threads, OpenMP) | 1.41-1.43 s |
| GPU, before image staging (`44caf37` + libclc) | 2.29 s |
| GPU, `c2e4e40` | 0.61 s |
| GPU, `51e56ca` (exact readback) | **0.61-0.65 s** |

GPU output is within 2/255 of the CPU export (mean difference ~0) and
bit-identical between GPU runs.

## 4. Loop overhead: counted loops ran at ~30% of FMA peak (`68451a0`)

`bench/unroll.c`: the same FMA work in a loop with four FMAs per iteration:

| loop | before | after |
|---|---:|---:|
| as written | 795 GFLOPS | 2029-2157 |
| `#pragma unroll 8` (ignored: Rusticl compiles at clang -O0) | 797 | 2114-2166 |
| constant trip count 4096 (too long to unroll fully) | 789 | 2129 |
| unrolled 4x by hand | 2201 | 2151-2311 |
| unrolled 8x by hand (not touched: body too big) | 2395 | 2362 |
| unrolled 4x with an exit check per copy | 958 | 959 |

The last row is why NIR's existing partial unroll (a check after every copy)
does not help: on AGX the check costs about as much as the back edge. New
pass `nir_opt_loop_unroll_runtime`: a main loop of K check-free copies
guarded by one "K iterations remain" test, the original loop kept as the
remainder. Only innermost loops with a single exit right after a side-effect
free header, a basic induction variable stepped by a positive constant, a
loop-invariant or constant bound and a `<` compare; bodies up to 32 cost
units (8x / 4x / 2x). `tests/loops.c` (signed/unsigned/64-bit counters,
steps 1-7, zero and short trip counts, bounds next to INT_MAX/UINT_MAX,
stores and ifs in the body, counter used after the loop, barriers; every
work-item with its own bounds) matches the CPU.

darktable's export is unchanged by it (demosaic 0.185 vs 0.187 s): its
kernels are not bound by small counted loops. The gain is for kernels that
are -- reductions, naive GEMM/convolution inner loops, iterative solvers.

## 5. Conformance

Khronos OpenCL CTS (`cts-run.sh`, `cts-compare.sh`, logs in `cts-results/`).

* `9e79757` against the same build with the old behaviour switched back on:
  failure sets identical (the same 4 pre-existing failures: `basic
  bufferreadwriterect`, `printf vector`, `commonfns sign`, and
  `non_uniform_work_group` exiting without a summary), no GPU faults.
* `817bbf4`, the full branch, 30 suites including all image suites and
  wimpy-mode math (`math_brute_force`, 106/106) and `conversions`: the
  shared suites' failure set is identical to the baseline above. The only
  other failures are 51 of 3433 `images/kernel_read_write` subtests (sRGB
  formats with linear filtering), and the stock system Mesa 26.1.8 fails
  exactly the same 51 format/filter/addressing combinations
  (`cts-results/system-26.1.8/`). No regressions, no GPU faults. `51e56ca`
  changes only the cache thresholds; the `basic` image tests pass on it.

## 6. Round-trip latency is the GPU firmware, not the driver

`bench/latency.c`, `bench/latprof.c`, kernel tracepoints
(`gpu_scheduler:*`, `irq:irq_handler_entry` for `206408000.mbox-recv`), one
tiny kernel + clFinish, median:

| GPU idle before submit | clFinish round trip | job_run -> firmware "done" IRQ |
|---|---:|---:|
| none (back-to-back) | 98 us | 58 us |
| 1 ms | 145-176 us | 89 us |
| 20 ms | 456 us | 300 us |

The kernel runs ~1.5 us on the GPU. DRM scheduler hop ~10 us, completion
handling ~8-40 us, Rusticl's thread hand-offs and fence export/import
~20 us; the rest is firmware: per-job overhead (~55 us) and power-up after
idle. The lever would be the firmware's idle power-off settings in the
kernel driver's init data, out of scope for a userspace driver. Workloads
that synchronise after every small kernel pay it; batching work between
synchronisation points avoids it, and fix 1 makes that batching effective.

## Limits and non-findings

* **2 GiB per allocation.** Rusticl clamps to `i32::MAX` because Gallium's
  transfer API uses signed 32-bit boxes; hashcat halves it again.
* **Maps still copy.** Rusticl maps through a malloc'd shadow, half the
  speed of ReadBuffer. A zero-copy map for write-back buffers is possible.
* **CL_MEM_USE_HOST_PTR** buffers always shadow-copy: Asahi cannot import
  user memory (no `resource_from_user_memory`, needs kernel support).
* **hashcat** (MD5 3.3 GH/s) is close to the integer ALU limit already, and
  32-bit rotates already compile to one `extr`. Nothing to gain there.

## Hazards (read before running anything)

* **GPU memory is invisible to the OOM killer.** Buffer objects are pinned
  shmem charged to the cgroup but not to the process, so an over-allocating
  test gets bystanders killed (here: a user's systemd manager and D-Bus).
  Run GPU tests with `capped.sh`; inside a capped scope only the test dies.
* **Killing a process with GPU jobs in flight** produced a GPU timeout with
  an unmapped CDM read and firmware recovery (system driver). Let
  benchmarks finish.
* **A/B with `AGX_MESA_DEBUG`** disables the shader disk cache on that side
  only; disable it on both.
* **Editing `cts-run.sh` while it runs** corrupts the run (bash reads
  scripts incrementally); replace it with `mv` instead.

## Next

* Run the Vulkan CTS subset (got-bringup) on a Honeykrisp built from this
  branch before moving `local-deploy`, or restrict the unroll pass to OpenCL.
* Zero-copy map for write-back buffers.
* Staged reads for arrays/3D/cubes; faster raw-integer staging blits.
* Other machines in the fleet: M1 Pro (G13S), M1 Max (G13C), M1 Ultra (G13D).
* Real workloads beyond darktable: OpenCV T-API, pyopencl, ffmpeg OpenCL
  filters, llama.cpp's OpenCL backend.
