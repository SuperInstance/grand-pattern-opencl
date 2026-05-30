/**
 * conservation.cl — Double-entry bookkeeping verification.
 *
 * Checks that perception and prediction counts are balanced
 * within a tolerance, enforcing conservation laws across the graph.
 */

__kernel void conservation_check(
    __global const int* perception_counts, // [N]
    __global const int* prediction_counts, // [N]
    const int tolerance,                   // allowed imbalance
    const int n,                           // room count
    __global int* violations               // [N] 1 if violated
) {
    int room_id = get_global_id(0);
    if (room_id >= n) return;

    int diff = abs(perception_counts[room_id] - prediction_counts[room_id]);
    violations[room_id] = (diff > tolerance) ? 1 : 0;
}
