# Grand Pattern — OpenCL GPU Implementation

GPU-accelerated cellular graph computation for the Grand Pattern. Parallel vibe diffusion, JEPA prediction, murmur gossip, signal routing, and tick processing — all running across thousands of rooms on OpenCL-capable hardware.

## Architecture

```
src/
  kernels/
    vibe_diffusion.cl     — Reaction-diffusion for 16-dim vibe embeddings
    jepa_predict.cl       — JEPA prediction + surprise (cosine distance)
    murmur_gossip.cl      — Parallel gossip propagation with TTL decay
    signal_route.cl       — Deadband signal routing
    tick_process.cl       — Perception window advancement + energy
    graph_ops.cl          — BFS traversal + anomaly detection
    conservation.cl       — Double-entry bookkeeping verification
    fleet_ops.cl          — Fleet-level vibe/surprise/energy reduction
  host/
    opencl_host.h         — C host API
    opencl_host.c         — Implementation (context, buffers, kernel dispatch)
tests/
  test_harness.c          — 20+ tests
```

## Kernels

| Kernel | Work-items | Purpose |
|--------|-----------|---------|
| `vibe_diffuse` | 1 per room | Weighted neighbor diffusion, clamped [-1,1] |
| `jepa_predict` | 1 per room | Average perception window → predict → cosine surprise |
| `murmur_propagate` | 1 per murmur | Blend gossip vibe into target, decay TTL |
| `signal_route` | 1 per signal | Route if L2 distance exceeds deadband |
| `tick_process` | 1 per room | Shift perception window, update energy |
| `bfs_step` | 1 per node | Single BFS frontier expansion |
| `anomaly_detect` | 1 per room | Flag high surprise + vibe divergence from fleet |
| `conservation_check` | 1 per room | Verify perception/prediction count balance |
| `fleet_vibe_reduce` | 1 per dim | Parallel reduction to fleet average |

## Build

```bash
# Install OpenCL headers (Ubuntu/Debian)
sudo apt install opencl-headers ocl-icd-opencl-dev

# Build library
make

# Run tests
make test
```

## Usage

```c
#include "host/opencl_host.h"

GrandPatternCL* cl = gpcl_init();
gpcl_create_graph(cl, 1000);

// Set edges
int from[] = {0, 1, ...};
int to[]   = {1, 0, ...};
double w[] = {1.0, 1.0, ...};
gpcl_set_edges(cl, from, to, w, edge_count);

// Set initial vibes
double vibes[1000 * 16] = {...};
gpcl_set_vibes(cl, vibes);

// Run ticks
for (int t = 0; t < 100; t++) {
    gpcl_tick(cl);  // diffuse → predict → tick → fleet reduce
}

// Query results
double fleet[16];
gpcl_get_fleet_vibe(cl, fleet);

gpcl_destroy(cl);
```

## Host API

### Lifecycle
- `gpcl_init()` — Create OpenCL context, compile all kernels
- `gpcl_destroy()` — Release all resources

### Graph Setup
- `gpcl_create_graph(cl, room_count)` — Allocate buffers
- `gpcl_set_edges(cl, from, to, weights, edge_count)` — Define graph topology
- `gpcl_set_vibes(cl, vibes)` — Set room vibe vectors
- `gpcl_set_murmurs(cl, vibes, surprise, targets, ttl, count)` — Load gossip messages
- `gpcl_set_conservation_data(cl, perception_counts, prediction_counts)` — Set bookkeeping data

### Operations
- `gpcl_tick(cl)` — Full tick cycle (diffuse → predict → tick → fleet reduce)
- `gpcl_diffuse_vibes(cl, coefficient)` — Single diffusion step
- `gpcl_predict_all(cl)` — JEPA prediction + surprise
- `gpcl_gossip(cl, blend_rate)` — Murmur propagation
- `gpcl_check_conservation(cl, tolerance)` — Verify conservation
- `gpcl_bfs(cl, source)` — BFS from source node
- `gpcl_detect_anomalies(cl, surprise_thresh, divergence_thresh)` — Find anomalous rooms

### Queries
- `gpcl_get_vibes`, `gpcl_get_surprise`, `gpcl_get_fleet_vibe`
- `gpcl_get_violations`, `gpcl_get_predicted`, `gpcl_get_energy`
- `gpcl_get_distances`, `gpcl_get_anomalies`

## Requirements

- OpenCL 1.2+ (GPU or CPU device)
- C11 compiler
- OpenCL ICD and headers

## License

MIT
