# Build configuration
BUILD_TYPE ?= release
DEBUG ?= 0

# Numeric mode for cuBLAS SGEMMs (shared by self-kNN and DC reduction).
#   USE_CUBLAS_TF32=1 -> TF32 tensor-core math (CUBLAS_TF32_TENSOR_OP_MATH),
#                       binary suffix "-TF32".
#   USE_CUBLAS_TF32=0 -> strict FP32 (CUBLAS_DEFAULT_MATH),
#                       binary suffix "-FP32".
USE_CUBLAS_TF32 ?= 0

# Toolchain and dependency locations (override on the command line, see README).
#   CUDA_PATH      CUDA toolkit root
#   HOST_COMPILER  host compiler passed to nvcc via -ccbin (empty: nvcc's default)
#   SMS            GPU compute capability, e.g. 70, 75, 80, 86, 89, 90
#   FAISS_PREFIX   FAISS install prefix, used with USE_FAISS=1 (default: the active conda environment)
#   USE_OPENBLAS=1 link OpenBLAS instead of MKL (match the BLAS your FAISS uses)
CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?=
C_STD ?= c++17
SMS ?= 86
FAISS_PREFIX ?= $(CONDA_PREFIX)
FAISS_INC_DIR ?= $(FAISS_PREFIX)/include
FAISS_LIB_DIR ?= $(FAISS_PREFIX)/lib
FAISS_CORE_LIBS ?= -lfaiss -lfaiss_avx2
ifeq ($(USE_OPENBLAS),1)
    BLAS_LIBS ?= -lopenblas
else
    BLAS_LIBS ?= -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lgomp -lpthread -ldl
endif

# Validate toolchains
ifeq ($(wildcard $(CUDA_PATH)/bin/nvcc),)
    $(error ERROR - CUDA not found at $(CUDA_PATH))
endif


# Compiler and linker flags
NVCC := $(CUDA_PATH)/bin/nvcc
ifneq ($(strip $(HOST_COMPILER)),)
    NVCC += -ccbin $(HOST_COMPILER)
endif


CFLAGS := -std=$(C_STD) -Wno-deprecated-declarations -I. -Ifull_gpu
LDFLAGS := -lcudart -lstdc++ -lm -lcublas -lcublasLt
DEPFLAGS := -MMD -MP

# Keep compatibility with code paths guarded by USE_CPP11.
ifeq ($(C_STD),c++11)
    CFLAGS += -D USE_CPP11
endif

# Generate CUDA compute capability flags
GENCODE_FLAGS := -gencode arch=compute_$(SMS),code=sm_$(SMS)
GENCODE_FLAGS += -gencode arch=compute_$(SMS),code=compute_$(SMS)

# Debug flags
ifeq ($(DEBUG),1)
    CFLAGS += -g -G
    BUILD_TYPE := debug
    HOST_OPT ?= -O0
endif

# Opt in with `make HOST_OPT=-O3` for host code optimization.
HOST_OPT ?= -O0  #-O3
CFLAGS += $(HOST_OPT)

# Opt in with `make USE_OPENMP=1` (requires a host compiler with OpenMP support).
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
    CFLAGS += -Xcompiler -fopenmp
endif

# Propagate the project-wide TF32 switch into both host and device code.
CFLAGS += -DUSE_CUBLAS_TF32=$(USE_CUBLAS_TF32)

# Fused distance selection (FDS) is disabled by default.
# Enable with `make USE_SKN_FDS=1` (optionally override SKN_FDS_MAX_DIM).
USE_SKN_FDS ?= 0
ifeq ($(USE_SKN_FDS),1)
    SKN_FDS_MAX_DIM ?= 128
else
    SKN_FDS_MAX_DIM := 0
endif
CFLAGS += -DSKN_FDS_MAX_DIM=$(SKN_FDS_MAX_DIM)

# Optional stage timing printouts from FastDECBase (kNN + DC reduction).
# Enable with `make FASTDEC_STAGE_TIMING=1`.
FASTDEC_STAGE_TIMING ?= 0
CFLAGS += -DFASTDEC_STAGE_TIMING=$(FASTDEC_STAGE_TIMING)

# Host threads for FastDECBase density estimation (only used at >= 100k points).
# Default 1; `make FASTDEC_DENSITY_THREADS=0` uses all cores. The env variable
# of the same name overrides this at run time. Run `make clean` when switching it.
FASTDEC_DENSITY_THREADS ?= 1
CFLAGS += -DFASTDEC_DENSITY_THREADS=$(FASTDEC_DENSITY_THREADS)

ifdef SKN_SELECTOR_WARPS_PER_BLOCK
    CFLAGS += -DSKN_SELECTOR_WARPS_PER_BLOCK=$(SKN_SELECTOR_WARPS_PER_BLOCK)
endif
ifdef SKN_COALESCED_TRANSPOSE_MIN_K
    CFLAGS += -DSKN_COALESCED_TRANSPOSE_MIN_K=$(SKN_COALESCED_TRANSPOSE_MIN_K)
endif
ifdef SKN_HEAP_HOLE_SIFT
    CFLAGS += -DSKN_HEAP_HOLE_SIFT=$(SKN_HEAP_HOLE_SIFT)
endif
ifdef SKN_FORCE_TILE
    CFLAGS += -DSKN_FORCE_TILE=$(SKN_FORCE_TILE)
endif
# Single-read reverse selection (default 1 in symtriknncuda.cu); 0 restores the strided reverse pass.
ifdef SKN_FUSED_REVERSE
    CFLAGS += -DSKN_FUSED_REVERSE=$(SKN_FUSED_REVERSE)
endif
# Reference mode for exactness checks: dense tiles of the fused distances everywhere (default 0).
ifdef SKN_FDS_REFERENCE
    CFLAGS += -DSKN_FDS_REFERENCE=$(SKN_FDS_REFERENCE)
endif
ifdef DC_CUBLAS_SINGLE_SLAB_MAX_MB
    CFLAGS += -DDC_CUBLAS_SINGLE_SLAB_MAX_MB=$(DC_CUBLAS_SINGLE_SLAB_MAX_MB)
endif

# Optional override of the dimensionality below which TF32 falls back to strict
# FP32 (default 8, defined in knndcshared.h), e.g. make USE_CUBLAS_TF32=1 TF32_MIN_DIM=16
ifdef TF32_MIN_DIM
    CFLAGS += -DTF32_MIN_DIM=$(TF32_MIN_DIM)
endif

# Density estimator used by FastDEC::Fit.
#   USE_GAUSSIAN_KERNEL=0 -> Laplacian kernel exp(-d/(2*d_bar))      (default)
#   USE_GAUSSIAN_KERNEL=1 -> Gaussian kernel  exp(-d^2/(2*d_bar^2))
# Object files do not track this flag: run `make clean` when switching it.
USE_GAUSSIAN_KERNEL ?= 0
CFLAGS += -DUSE_GAUSSIAN_KERNEL=$(USE_GAUSSIAN_KERNEL)

# Optional kNN-CUDA baseline backend (`--backend knncuda`). Off by default because
# knncuda-fastdec/ is derived from kNN-CUDA and licensed CC BY-NC-SA 3.0, not MIT
# (see knncuda-fastdec/LICENSE). Enable with `make USE_KNNCUDA=1`; run `make clean`
# when switching it.
USE_KNNCUDA ?= 0
CFLAGS += -DUSE_KNNCUDA=$(USE_KNNCUDA)
ifeq ($(USE_KNNCUDA),1)
    CFLAGS += -Iknncuda-fastdec
endif

# Optional FAISS GPU baseline backend (`--backend faissgpu`). Off by default so the
# default build has no FAISS/BLAS dependency. Enable with `make USE_FAISS=1`
# (set FAISS_PREFIX if FAISS is not in the active conda environment); run
# `make clean` when switching it.
USE_FAISS ?= 0
CFLAGS += -DUSE_FAISS=$(USE_FAISS)
ifeq ($(USE_FAISS),1)
    # Only the main executable needs FAISS; the tests and clean targets do not.
    NO_FAISS_GOALS := clean clobber test-sort test-unit sort_topk_test symwarp_unit_test
    ifneq ($(if $(MAKECMDGOALS),$(filter-out $(NO_FAISS_GOALS),$(MAKECMDGOALS)),all),)
        ifeq ($(wildcard $(FAISS_INC_DIR)/faiss/gpu/GpuIndexFlat.h),)
            $(error ERROR - FAISS GPU headers not found under $(FAISS_INC_DIR); set FAISS_PREFIX=/path/to/faiss)
        endif
    endif
    CFLAGS += -Ifaissgpu-fastdec -I$(FAISS_INC_DIR)
    LDFLAGS += -L$(FAISS_LIB_DIR) $(FAISS_CORE_LIBS) $(BLAS_LIBS) -Xlinker -rpath -Xlinker $(FAISS_LIB_DIR)
endif

# Pick the binary suffix from the TF32 setting.
ifeq ($(USE_CUBLAS_TF32),1)
    TF32_SUFFIX := TF32
else
    TF32_SUFFIX := FP32
endif

# Print build configuration
$(info >>> INFO: Build Type = $(BUILD_TYPE) <<<)
$(info >>> INFO: Compute Capability = $(SMS) <<<)
$(info >>> INFO: CUDA Path = $(CUDA_PATH) <<<)
$(info >>> INFO: Host Compiler = $(if $(strip $(HOST_COMPILER)),$(HOST_COMPILER),nvcc default) <<<)
$(info >>> INFO: C++ Standard = $(C_STD) <<<)
$(info >>> INFO: USE_CUBLAS_TF32 = $(USE_CUBLAS_TF32) ($(TF32_SUFFIX)) <<<)
$(info >>> INFO: USE_SKN_FDS = $(USE_SKN_FDS), SKN_FDS_MAX_DIM = $(SKN_FDS_MAX_DIM) <<<)
$(info >>> INFO: FASTDEC_STAGE_TIMING = $(FASTDEC_STAGE_TIMING) <<<)
$(info >>> INFO: FASTDEC_DENSITY_THREADS = $(FASTDEC_DENSITY_THREADS) <<<)
$(info >>> INFO: USE_GAUSSIAN_KERNEL = $(USE_GAUSSIAN_KERNEL) <<<)
$(info >>> INFO: USE_KNNCUDA = $(USE_KNNCUDA) <<<)
$(info >>> INFO: USE_FAISS = $(USE_FAISS)$(if $(filter 1,$(USE_FAISS)), (include dir: $(FAISS_INC_DIR))) <<<)
$(info >>> INFO: Host optimization = $(HOST_OPT), USE_OPENMP = $(USE_OPENMP) <<<)

################################################################################
# Build rules
################################################################################

KNNCUDA_OBJECTS := knncuda-fastdec/knncudafastdec.o knncuda-fastdec/knncuda.o
FAISS_OBJECTS := faissgpu-fastdec/faissgpufastdec.o faissgpu-fastdec/faissknncuda.o
OBJECTS := test.o fastdecbase.o dataset.o symwarpfastdec.o helper.o symtriknncuda.o dcreductioncuda.o full_gpu/symwarpfullgpufastdec.o full_gpu/symwarpfastdeccuda.o
ifeq ($(USE_KNNCUDA),1)
    OBJECTS += $(KNNCUDA_OBJECTS)
endif
ifeq ($(USE_FAISS),1)
    OBJECTS += $(FAISS_OBJECTS)
endif
EXECUTABLE := symwarp-fastdec

UNIT_TEST_OBJECTS := symwarp_unit_test.o fastdecbase.o dataset.o symwarpfastdec.o helper.o symtriknncuda.o dcreductioncuda.o
UNIT_TEST_EXECUTABLE := symwarp_unit_test

.PHONY: all build run test test-sort test-unit clean clobber

all: build

build: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@echo "Linking $(EXECUTABLE)..."
	$(NVCC) -m64 $(CFLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Build successful!"
	@echo "  Executable: $(shell pwd)/$(EXECUTABLE)"

%.o: %.cpp
	@echo "Compiling $<..."
	$(NVCC) -m64 $(CFLAGS) $(DEPFLAGS) $(GENCODE_FLAGS) -c $< -o $@

%.o: %.cu
	@echo "Compiling $<..."
	$(NVCC) -m64 $(CFLAGS) $(DEPFLAGS) $(GENCODE_FLAGS) -c $< -o $@

run: build
	@./$(EXECUTABLE)

sort_topk_test: sort_topk_test.cu symtriknncuda.cu symtriknncuda.h knndcshared.h dataset.cpp dataset.h
	$(NVCC) -m64 $(CFLAGS) $(GENCODE_FLAGS) -o $@ sort_topk_test.cu dataset.cpp $(LDFLAGS)

test-sort: sort_topk_test
	./sort_topk_test

$(UNIT_TEST_EXECUTABLE): $(UNIT_TEST_OBJECTS)
	$(NVCC) -m64 $(CFLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LDFLAGS)

test-unit: $(UNIT_TEST_EXECUTABLE)
	./$(UNIT_TEST_EXECUTABLE)

test: test-sort test-unit

clean:
	@rm -f $(OBJECTS) $(OBJECTS:.o=.d) $(KNNCUDA_OBJECTS) $(KNNCUDA_OBJECTS:.o=.d) $(FAISS_OBJECTS) $(FAISS_OBJECTS:.o=.d) $(EXECUTABLE) sort_topk_test $(UNIT_TEST_EXECUTABLE) symwarp_unit_test.o
	@echo "✓ Clean successful!"
	@echo "  Removed: $(shell pwd)/$(EXECUTABLE)"

clobber: clean

-include $(OBJECTS:.o=.d)
