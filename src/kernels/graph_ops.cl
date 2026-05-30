/**
 * graph_ops.cl — Graph traversal, BFS, and anomaly detection.
 *
 * Provides parallel BFS distance computation and anomaly detection
 * based on surprise and vibe divergence across the graph.
 */

#define VIBE_DIM 16

// BFS: compute shortest hop distance from a source to all nodes.
// Runs iteratively; call multiple times until frontier is empty.
__kernel void bfs_step(
    __global const int* neighbor_offsets,  // [N+1] CSR row pointers
    __global const int* neighbor_indices,  // [M] neighbor indices
    __global int* distances,               // [N] current distances (-1 = unvisited)
    __global int* frontier,                // [N] 1 if in current frontier
    __global int* next_frontier,           // [N] output frontier
    const int n                            // node count
) {
    int node = get_global_id(0);
    if (node >= n) return;

    next_frontier[node] = 0;

    if (!frontier[node]) return;

    int start = neighbor_offsets[node];
    int end   = neighbor_offsets[node + 1];
    int my_dist = distances[node];

    for (int e = start; e < end; e++) {
        int nb = neighbor_indices[e];
        if (distances[nb] == -1 || distances[nb] > my_dist + 1) {
            distances[nb] = my_dist + 1;
            next_frontier[nb] = 1;
        }
    }
}

// Anomaly detection: flag rooms whose surprise exceeds a threshold
// AND whose vibe diverges significantly from the fleet average.
__kernel void anomaly_detect(
    __global const double* room_vibes,     // [N x VIBE_DIM]
    __global const double* room_surprise,  // [N]
    __global const double* fleet_vibe,     // [VIBE_DIM]
    const double surprise_threshold,
    const double divergence_threshold,
    const int n,
    __global int* anomalies                // [N] 1 if anomaly
) {
    int room_id = get_global_id(0);
    if (room_id >= n) return;

    double surprise = room_surprise[room_id];
    if (surprise <= surprise_threshold) {
        anomalies[room_id] = 0;
        return;
    }

    // Compute L2 distance from fleet vibe
    double dist_sq = 0.0;
    for (int d = 0; d < VIBE_DIM; d++) {
        double diff = room_vibes[room_id * VIBE_DIM + d] - fleet_vibe[d];
        dist_sq += diff * diff;
    }

    anomalies[room_id] = (sqrt(dist_sq) > divergence_threshold) ? 1 : 0;
}
