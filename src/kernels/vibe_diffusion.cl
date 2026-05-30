/**
 * vibe_diffusion.cl — Reaction-diffusion for 16-dimensional vibe embeddings.
 *
 * Each work-item updates one room's vibe vector by blending weighted
 * differences from neighbors (heat-equation style diffusion).
 * Results are clamped to [-1, 1] per dimension.
 */

#define VIBE_DIM 16

__kernel void vibe_diffuse(
    __global const double* vibes,          // [N x VIBE_DIM] current room vibes
    __global const double* neighbor_vibes, // [M x VIBE_DIM] neighbor vibe values
    __global const int* neighbor_offsets,  // [N+1] CSR row pointers
    __global const double* weights,        // [M] edge weights
    const double diffusion_coeff,          // diffusion strength
    __global double* out_vibes             // [N x VIBE_DIM] updated vibes
) {
    int room_id = get_global_id(0);
    int n = get_global_size(0);

    if (room_id >= n) return;

    int start = neighbor_offsets[room_id];
    int end   = neighbor_offsets[room_id + 1];

    double current[VIBE_DIM];
    for (int d = 0; d < VIBE_DIM; d++) {
        current[d] = vibes[room_id * VIBE_DIM + d];
    }

    // Accumulate weighted neighbor influence per dimension
    double delta[VIBE_DIM];
    for (int d = 0; d < VIBE_DIM; d++) {
        delta[d] = 0.0;
    }

    for (int e = start; e < end; e++) {
        double w = weights[e];
        for (int d = 0; d < VIBE_DIM; d++) {
            double nv = neighbor_vibes[e * VIBE_DIM + d];
            delta[d] += w * (nv - current[d]);
        }
    }

    // Apply diffusion and clamp
    for (int d = 0; d < VIBE_DIM; d++) {
        double updated = current[d] + diffusion_coeff * delta[d];
        // Clamp to [-1, 1]
        updated = fmax(-1.0, fmin(1.0, updated));
        out_vibes[room_id * VIBE_DIM + d] = updated;
    }
}
