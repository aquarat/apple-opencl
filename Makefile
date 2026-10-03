# Build the benchmarks and tests: plain C against the system ICD loader.
#   make            (needs ocl-icd-devel and opencl-headers)
CFLAGS ?= -O2 -Wall -Wno-unused-result
LDLIBS = -lOpenCL
BENCH = $(patsubst %.c,%,$(wildcard bench/*.c))
TESTS = $(patsubst %.c,%,$(wildcard tests/*.c))

all: $(BENCH) $(TESTS)
clean:
	rm -f $(BENCH) $(TESTS)
.PHONY: all clean
