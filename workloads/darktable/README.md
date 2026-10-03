# darktable export on the GPU

The raw file is not committed. It is `_DSC0009.ARW`, a Sony ILCE-7M3 sample
from [raw.pixls.us](https://raw.pixls.us/) (CC0):

    curl -L -o sample.ARW https://raw.pixls.us/data/Sony/ILCE-7M3/_DSC0009.ARW

Run darktable 5.6.1's default export pipeline on the local driver, with its
own configuration directory so your darktable settings are untouched:

    ../../with-cl.sh darktable-cli sample.ARW gpu.png --core --configdir $PWD/cfg \
        -d perf -d opencl --conf opencl=TRUE

and on the CPU for comparison:

    darktable-cli sample.ARW cpu.png --core --configdir $PWD/cfg-cpu -d perf --conf opencl=FALSE

`pixel pipeline processing took` is the figure quoted in STATE.md. Run it
more than once: the first GPU run also compiles darktable's kernels.
