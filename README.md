# Grand Pattern — OpenCL Implementation

**OpenCL 3.0 kernels** for the Fibonacci Dual-Direction Architecture.

Portable across GPU vendors (NVIDIA, AMD, Intel, Apple). Kernel code in `.cl` files, host in C.

## Architecture

The GPU kernels handle the parallel compute-heavy operations:
- **Embedding operations**: cosine similarity, centroid, distance — massively parallel
- **JEPA prediction**: batch prediction across all rooms simultaneously
- **Double-entry balance**: parallel reduction to check all rooms balance
- **GC merge**: parallel similarity computation → merge pairs
- **Vibe computation**: parallel reduction across perception/prediction DBs
- **Cross-room correlation**: matrix of all-pairs cosine similarity

## Kernels

| Kernel | Description |
|--------|-------------|
| `cosine_similarity` | Parallel cosine similarity with work-group reduction |
| `batch_predict` | Per-room JEPA prediction |
| `balance_check` | Parallel double-entry balance verification |
| `decay` | Element-wise exponential decay on strengths |
| `vibe_compute` | Parallel vibe computation + normalization |
| `correlation_matrix` | All-pairs cosine similarity matrix |
| `merge_candidates` | Identify merge candidates above threshold |

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Requires:
- OpenCL 3.0 headers and ICD loader
- A GPU with OpenCL support (NVIDIA, AMD, Intel, or Apple)
- CMake 3.20+

## Running Tests

```bash
./build/test_opencl_kernels
```

## Implementation Details

- **Work-group level reductions** using local memory (barrier + sequential reduce)
- **Portable kernel code** — no vendor-specific intrinsics
- **Coalesced global memory access** patterns
- **Atomic operations** for balance checking and merge candidate counting

## License

MIT
