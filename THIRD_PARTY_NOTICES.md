# Third-Party Notices

This project is licensed under the MIT License (see `LICENSE`), except where noted below.

## Code included in this repository

### kNN-CUDA (optional baseline backend)

- **Location:** `knncuda-fastdec/`
- **Upstream:** https://github.com/vincentfpgarcia/kNN-CUDA
- **Authors:** Vincent Garcia, Éric Debreuve, Michel Barlaud
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0); see `knncuda-fastdec/LICENSE`
- **Modifications:** We extended the global-memory implementation with tiled/batched processing for large datasets, added alternative top-k selection kernels, and wrote a FastDEC integration layer.
- **Build:** Not compiled by default. It is only built with `make USE_KNNCUDA=1`. Binaries built that way include CC BY-NC-SA 3.0 code and **may not be used for commercial purposes**.

References:
- V. Garcia, E. Debreuve, F. Nielsen, M. Barlaud. *k-nearest neighbor search: fast GPU-based implementations and application to high-dimensional feature matching.* ICIP 2010.
- V. Garcia, E. Debreuve, M. Barlaud. *Fast k nearest neighbor search using GPU.* CVPR Workshop on Computer Vision on GPU, 2008.

## Libraries linked at build time (not redistributed)

| Library | Used for | License |
|---|---|---|
| [FAISS](https://github.com/facebookresearch/faiss) 1.12.0 | `faissgpu` baseline backend, only with `make USE_FAISS=1` | MIT |
| NVIDIA CUDA Toolkit (CUDA runtime, cuBLAS, cuBLASLt) | GPU kernels, GEMMs | [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/) |
| Intel oneMKL *or* OpenBLAS (pulled in by FAISS) | BLAS for FAISS, only with `make USE_FAISS=1` | [Intel Simplified Software License](https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html) / BSD-3-Clause |

These libraries are not part of this repository. Each is covered by its own license when you install it.

## Data

### Spiral (`spiral.csv`)

312 two-dimensional points in 3 clusters, normalized to [0, 1]. It is a widely used clustering benchmark from:

- H. Chang, D.-Y. Yeung. *Robust path-based spectral clustering.* Pattern Recognition 41(1):191–203, 2008.

It is included only as a small example input.

### GloVe (not included)

The large-scale experiments use the 2024 Wikipedia + Gigaword GloVe vectors (`glove.2024.wikigiga.50d.zip`) from https://nlp.stanford.edu/projects/glove/. The vectors are not redistributed here; `scripts/prepare_glove.py` converts the downloaded file into the CSV format this program reads. See the GloVe project page for the vectors' license and citation.

- J. Pennington, R. Socher, C. D. Manning. *GloVe: Global Vectors for Word Representation.* EMNLP 2014.

## Algorithm

This project accelerates FastDEC; the original Python implementation is not included:

- G. Yang, H. Lv, Y. Yang, Z. Gong, X. Chen, Z. Hao. *FastDEC: Clustering by Fast Dominance Estimation.* ECML-PKDD 2022, LNCS, pp. 138–156, Springer, 2023. https://doi.org/10.1007/978-3-031-26387-3_9
