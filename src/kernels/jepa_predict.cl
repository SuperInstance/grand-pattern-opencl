/**
 * jepa_predict.cl — JEPA prediction and surprise computation.
 *
 * Each work-item predicts a room's next state by averaging a sliding
 * window of perception history, then computes surprise as cosine
 * distance between prediction and actual.
 */

__kernel void jepa_predict(
    __global const double* perception_db,  // [N x max_window x D]
    __global const double* actual_db,      // [N x D] current actual state
    __global const int* window_sizes,      // [N] per-room window length
    const int max_window,                  // max window size (stride)
    const int dim,                         // dimensionality
    __global double* predicted,            // [N x D] output predictions
    __global double* surprise              // [N] output surprise scores
) {
    int room_id = get_global_id(0);
    int n = get_global_size(0);
    if (room_id >= n) return;

    int ws = window_sizes[room_id];
    if (ws <= 0) {
        surprise[room_id] = 0.0;
        for (int d = 0; d < dim; d++) {
            predicted[room_id * dim + d] = 0.0;
        }
        return;
    }

    // Compute prediction: average of perception window
    double pred[16]; // max dim supported in private memory
    for (int d = 0; d < dim; d++) {
        pred[d] = 0.0;
    }

    for (int t = 0; t < ws; t++) {
        for (int d = 0; d < dim; d++) {
            pred[d] += perception_db[room_id * max_window * dim + t * dim + d];
        }
    }

    for (int d = 0; d < dim; d++) {
        pred[d] /= (double)ws;
        predicted[room_id * dim + d] = pred[d];
    }

    // Compute cosine distance between prediction and actual
    double dot_pp = 0.0, dot_aa = 0.0, dot_pa = 0.0;
    for (int d = 0; d < dim; d++) {
        double a = actual_db[room_id * dim + d];
        dot_pa += pred[d] * a;
        dot_pp += pred[d] * pred[d];
        dot_aa += a * a;
    }

    double norm_p = sqrt(dot_pp);
    double norm_a = sqrt(dot_aa);
    double cosine_sim = 0.0;
    if (norm_p > 1e-12 && norm_a > 1e-12) {
        cosine_sim = dot_pa / (norm_p * norm_a);
    }
    // Cosine distance: 1 - cosine_similarity, clamped
    double cd = 1.0 - cosine_sim;
    surprise[room_id] = fmax(0.0, fmin(2.0, cd));
}
