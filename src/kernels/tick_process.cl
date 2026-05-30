/**
 * tick_process.cl — Tick processing across all rooms.
 *
 * Advances each room one tick: ages perceptions, shifts windows,
 * and accumulates energy from neighbor activity.
 */

#define VIBE_DIM 16

__kernel void tick_process(
    __global double* perception_db,        // [N x max_window x D] perception history
    __global int* window_sizes,            // [N] current window fill
    const int max_window,                  // max window size
    const int dim,                         // dimensionality
    __global const double* current_vibes,  // [N x D] current vibes to append
    __global double* energy,               // [N] room energy levels
    __global const double* surprise,       // [N] surprise scores
    const double energy_decay              // energy decay per tick
) {
    int room_id = get_global_id(0);
    int n = get_global_size(0);
    if (room_id >= n) return;

    int ws = window_sizes[room_id];
    int base = room_id * max_window * dim;

    // Shift perception window left by 1 (drop oldest if full)
    if (ws >= max_window) {
        // Shift: move [1..max_window-1] to [0..max_window-2]
        for (int t = 0; t < max_window - 1; t++) {
            for (int d = 0; d < dim; d++) {
                perception_db[base + t * dim + d] =
                    perception_db[base + (t + 1) * dim + d];
            }
        }
        // Append current vibe at last position
        for (int d = 0; d < dim; d++) {
            perception_db[base + (max_window - 1) * dim + d] =
                current_vibes[room_id * dim + d];
        }
    } else {
        // Not full yet — just append
        for (int d = 0; d < dim; d++) {
            perception_db[base + ws * dim + d] =
                current_vibes[room_id * dim + d];
        }
        window_sizes[room_id] = ws + 1;
    }

    // Update energy: decay + surprise injection
    double e = energy[room_id];
    e = e * energy_decay + surprise[room_id] * 0.1;
    energy[room_id] = fmax(0.0, fmin(10.0, e));
}
