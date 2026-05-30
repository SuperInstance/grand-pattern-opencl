/**
 * fleet_ops.cl — Fleet-level aggregation operations.
 *
 * Reduces room-level data (vibes, surprise, energy) to fleet-level
 * summaries via parallel reduction.
 */

#define VIBE_DIM 16

// Average all room vibes into a single fleet vibe vector
__kernel void fleet_vibe_reduce(
    __global const double* room_vibes,     // [N x VIBE_DIM]
    const int room_count,
    __global double* fleet_vibe            // [VIBE_DIM]
) {
    int dim = get_global_id(0);
    if (dim >= VIBE_DIM) return;

    double sum = 0.0;
    for (int i = 0; i < room_count; i++) {
        sum += room_vibes[i * VIBE_DIM + dim];
    }
    fleet_vibe[dim] = sum / (double)room_count;
}

// Compute fleet-wide average surprise
__kernel void fleet_surprise_reduce(
    __global const double* room_surprise,  // [N]
    const int room_count,
    __global double* fleet_surprise        // [1]
) {
    int tid = get_global_id(0);
    if (tid != 0) return;

    double sum = 0.0;
    for (int i = 0; i < room_count; i++) {
        sum += room_surprise[i];
    }
    fleet_surprise[0] = sum / (double)room_count;
}

// Compute fleet-wide total energy
__kernel void fleet_energy_reduce(
    __global const double* room_energy,    // [N]
    const int room_count,
    __global double* fleet_energy          // [1]
) {
    int tid = get_global_id(0);
    if (tid != 0) return;

    double sum = 0.0;
    for (int i = 0; i < room_count; i++) {
        sum += room_energy[i];
    }
    fleet_energy[0] = sum;
}
