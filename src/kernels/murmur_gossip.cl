/**
 * murmur_gossip.cl — Parallel murmur/gossip propagation.
 *
 * Each work-item handles one murmur: blends its vibe into the target
 * room and decays TTL. Murmurs are gossip messages that propagate
 * vibe information across the graph.
 */

#define VIBE_DIM 16

__kernel void murmur_propagate(
    __global const double* murmur_vibes,   // [M x VIBE_DIM] murmur vibe payloads
    __global const double* murmur_surprise, // [M] murmur surprise values
    __global const int* targets,            // [M] target room IDs
    __global const int* ttl,                // [M] time-to-live counters
    const double blend_rate,                // how strongly murmur blends
    __global double* room_vibes,            // [N x VIBE_DIM] updated room vibes
    __global double* room_surprise,         // [N] updated room surprise
    __global int* murmur_ttl_out            // [M] updated TTL
) {
    int murmur_id = get_global_id(0);
    int m = get_global_size(0);
    if (murmur_id >= m) return;

    int cur_ttl = ttl[murmur_id];
    if (cur_ttl <= 0) {
        murmur_ttl_out[murmur_id] = 0;
        return;
    }

    int target = targets[murmur_id];
    double ms = murmur_surprise[murmur_id];

    // Blend murmur vibe into target room
    // Weight decreases as TTL decays
    double ttl_factor = (double)cur_ttl / 10.0;
    ttl_factor = fmin(1.0, ttl_factor);
    double rate = blend_rate * ttl_factor;

    for (int d = 0; d < VIBE_DIM; d++) {
        double mv = murmur_vibes[murmur_id * VIBE_DIM + d];
        double rv = room_vibes[target * VIBE_DIM + d];
        double blended = rv + rate * (mv - rv);
        blended = fmax(-1.0, fmin(1.0, blended));
        room_vibes[target * VIBE_DIM + d] = blended;
    }

    // Blend surprise into target
    double rs = room_surprise[target];
    room_surprise[target] = rs + rate * 0.5 * (ms - rs);

    // Decay TTL
    murmur_ttl_out[murmur_id] = cur_ttl - 1;
}
