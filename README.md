The source code for the "Accelerating FastDEC Clustering Algorithm with GPUs" paper.

## Requirements

- Linux x86_64 with an NVIDIA GPU (compute capability 7.0 or newer, 8.0+ for TF32; set `SMS` to match). Only 8.6 has been tested.
- CUDA Toolkit 12.x (`nvcc`, cuBLAS, cuBLASLt)
- A C++17 host compiler supported by your CUDA version (e.g. GCC 9–12 for CUDA 12.4)
- Optional, only for the `faissgpu` baseline (`make USE_FAISS=1`): FAISS with GPU support (headers + `libfaiss`/`libfaiss_avx2`) and the MKL or OpenBLAS it was built with
- Python 3 with numpy and scikit-learn (only to prepare the GloVe dataset)

Developed and benchmarked with CUDA 12.4, GCC 11.5, FAISS 1.12.0, and an NVIDIA RTX A6000 (driver 565.57.01).

## Build

Build from this directory. The default build needs only the CUDA Toolkit and provides the `symwarp` and `fullgpu` backends:

```bash
make build
make build SMS=80                            # target another GPU (80 = A100, 89 = L40/RTX 4090, 90 = H100)
make build CUDA_PATH=/usr/local/cuda-12.4 HOST_COMPILER=g++-11
```

### Optional baseline backends

The two kNN baselines from the paper are off by default. Enable either or both at build time:

| Backend | Enable with | Extra requirement |
|---|---|---|
| `knncuda` | `make build USE_KNNCUDA=1` | None. Licensed CC BY-NC-SA 3.0 (non-commercial), see [License](#license) |
| `faissgpu` | `make build USE_FAISS=1` | FAISS with GPU support |

For `USE_FAISS=1`, install FAISS into a conda environment, for example:

```bash
conda create -n faissgpu -c pytorch -c nvidia faiss-gpu=1.12.0
conda activate faissgpu
make build USE_FAISS=1                            # uses the active conda env for FAISS
make build USE_FAISS=1 FAISS_PREFIX=/opt/faiss    # or point at any FAISS install
make build USE_FAISS=1 USE_OPENBLAS=1             # FAISS built against OpenBLAS instead of MKL
```

Selecting a backend that was not built in exits with an error naming the flag to rebuild with.

The build variants used in the paper:

- Default build (FP32, FDS disabled): `make build`
- Enable FDS for FP32 builds: `make build USE_SKN_FDS=1`
- TF32 build: `make build USE_CUBLAS_TF32=1`

Run `make clean` before switching any of these options; object files do not track them.

### Build options

| Option | Default | Description |
|---|---|---|
| `CUDA_PATH` | `/usr/local/cuda` | CUDA Toolkit root |
| `HOST_COMPILER` | nvcc default | Host compiler passed via `-ccbin` |
| `SMS` | `86` | GPU compute capability |
| `USE_KNNCUDA` | `0` | Build the `knncuda` baseline backend (CC BY-NC-SA 3.0, see [License](#license)) |
| `USE_FAISS` | `0` | Build the `faissgpu` baseline backend (requires FAISS) |
| `FAISS_PREFIX` | `$CONDA_PREFIX` | With `USE_FAISS=1`: FAISS install prefix (or set `FAISS_INC_DIR` / `FAISS_LIB_DIR`) |
| `USE_OPENBLAS` | `0` | With `USE_FAISS=1`: link OpenBLAS instead of MKL (or set `BLAS_LIBS` directly) |
| `USE_CUBLAS_TF32` | `0` | TF32 tensor-core GEMMs |
| `USE_SKN_FDS` | `0` | Fused distance selection in the self-kNN |
| `USE_GAUSSIAN_KERNEL` | `0` | Gaussian instead of Laplacian density kernel |
| `HOST_OPT` | `-O0` | Host optimization flags (the reported timings use the default) |
| `FASTDEC_DENSITY_THREADS` | `1` | Host threads for density estimation (`0` = all cores; only used at ≥100k points). The environment variable of the same name overrides it at run time. |
| `FASTDEC_STAGE_TIMING` | `0` | Print per-stage timings |

## Run

Run with a config file:

```bash
./symwarp-fastdec /path/to/config.txt
```

Select backend explicitly (all use the same config format):

```bash
./symwarp-fastdec --backend symwarp /path/to/config.txt
./symwarp-fastdec --backend knncuda /path/to/config.txt   # requires make USE_KNNCUDA=1
./symwarp-fastdec --backend faissgpu /path/to/config.txt  # requires make USE_FAISS=1
./symwarp-fastdec --backend fullgpu /path/to/config.txt
```

Config format:
1. dataset CSV path
2. k
3. cluster count
4. run count

## Datasets

- **Spiral:** included as `spiral.csv`.
- **GloVe 50d:** download `glove.2024.wikigiga.50d.zip` from https://github.com/stanfordnlp/glove, unzip it, and convert it (1,291,147 points, scaled to [-1, 1]):

  ```bash
  python scripts/prepare_glove.py <unzipped 50d vectors .txt> data/
  # -> data/glove_1291147_50d.csv
  ```

## Reproducibility commands

Spiral (FP32, FDS off):

```bash
cat > /tmp/spiral.cfg <<'EOF'
spiral.csv
7
3
1
EOF
make clean && make build USE_CUBLAS_TF32=0 USE_SKN_FDS=0
./symwarp-fastdec /tmp/spiral.cfg
```

Expected centers:

```text
[101,304,203]
```

GloVe 50d (FP32, FDS off):

```bash
cat > /tmp/glove50.cfg <<'EOF'
data/glove_1291147_50d.csv
45
20
1
EOF
make clean && make build USE_CUBLAS_TF32=0 USE_SKN_FDS=0
./symwarp-fastdec /tmp/glove50.cfg
```

Expected centers:

```text
[1238467,25582,1275690,1246285,1247244,1250707,34210,26718,25303,973395,32452,9915,1173687,120385,2835,1038526,103813,20050,1113047,12687]
```

GloVe 50d, k=45, c=20, 4 runs (excluding the first run) on a NVIDIA RTX A6000 GPU (the knncuda and faissgpu rows need `make USE_KNNCUDA=1 USE_FAISS=1`):

| Backend | Total (s) | kNN (s) | DC Reduction (s) | CV |
|---|---:|---:|---:|---:|
| knncuda | 340.68 | 339.49 | 0.39 | 0.08% |
| faissgpu | 35.76 | 35.14 | 0.035 | 0.15% |
| symwarp | 16.49 | 15.7 | 0.035 | 0.14% |

When FDS is on:

| Backend | Total (s) | kNN (s) | DC Reduction (s) | CV |
|---|---:|---:|---:|---:|
| symwarp | 10.91 | 8.71 | 0.08 | 0.12% |

  
GloVe 50d (TF32, FDS off):

```bash
make clean && make build USE_CUBLAS_TF32=1 USE_SKN_FDS=0
./symwarp-fastdec /tmp/glove50.cfg
```

Expected centers:

```text
[837605,25582,1259747,1246285,1247244,1250707,34210,26718,25303,973395,32452,9915,1173687,120385,2835,1038526,103813,20050,1113047,12687]
```

| Backend | Total (s) | kNN (s) | DC Reduction (s) | CV |
|---|---:|---:|---:|---:|
| symwarp | 15.24 | 13.15 | 0.036 | 0.10% 
  


## Tests

- Restored self-kNN unit test: `make test-sort`
- Additional unit tests (`SymWarpFastDEC::Fit` + `dc_reduction_cuda`): `make test-unit`
- All tests: `make test`

## License

This project is released under the [MIT License](LICENSE), except for the optional kNN-CUDA baseline in `knncuda-fastdec/`. That code is adapted from [kNN-CUDA](https://github.com/vincentfpgarcia/kNN-CUDA) and licensed under CC BY-NC-SA 3.0 (non-commercial). It is only compiled with `make USE_KNNCUDA=1`; binaries built that way must not be used commercially. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for all third-party components and datasets.

## Citation

If you use this code, please cite the paper; see [CITATION.cff](CITATION.cff).
