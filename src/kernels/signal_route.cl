/**
 * signal_route.cl — Signal routing with deadband.
 *
 * Routes signals between rooms based on vibe difference exceeding
 * a deadband threshold. Prevents thrashing by only propagating
 * signals when the change is significant.
 */

#define VIBE_DIM 16

__kernel void signal_route(
    __global const double* vibes,          // [N x VIBE_DIM]
    __global const int* src_rooms,         // [S] source rooms
    __global const int* dst_rooms,         // [S] destination rooms
    __global const double* deadband,       // [S] per-signal deadband threshold
    const int signal_count,                // total signals
    __global double* signal_out,           // [S x VIBE_DIM] routed signal values
    __global int* signal_active            // [S] 1 if signal was routed
) {
    int sig_id = get_global_id(0);
    if (sig_id >= signal_count) return;

    int src = src_rooms[sig_id];
    int dst = dst_rooms[sig_id];
    double db = deadband[sig_id];

    // Compute L2 distance between src and dst vibes
    double dist_sq = 0.0;
    for (int d = 0; d < VIBE_DIM; d++) {
        double diff = vibes[src * VIBE_DIM + d] - vibes[dst * VIBE_DIM + d];
        dist_sq += diff * diff;
    }
    double dist = sqrt(dist_sq);

    if (dist > db) {
        // Route: copy src vibe to signal output
        for (int d = 0; d < VIBE_DIM; d++) {
            signal_out[sig_id * VIBE_DIM + d] = vibes[src * VIBE_DIM + d];
        }
        signal_active[sig_id] = 1;
    } else {
        // Below deadband — no signal
        for (int d = 0; d < VIBE_DIM; d++) {
            signal_out[sig_id * VIBE_DIM + d] = 0.0;
        }
        signal_active[sig_id] = 0;
    }
}
