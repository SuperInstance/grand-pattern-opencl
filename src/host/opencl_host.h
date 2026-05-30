/**
 * opencl_host.h — C host API for Grand Pattern OpenCL kernels.
 */

#ifndef GRAND_PATTERN_CL_H
#define GRAND_PATTERN_CL_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stddef.h>

#define GP_VIBE_DIM 16
#define GP_MAX_WINDOW 32

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    /* Kernel handles */
    cl_kernel k_vibe_diffuse;
    cl_kernel k_jepa_predict;
    cl_kernel k_murmur_propagate;
    cl_kernel k_signal_route;
    cl_kernel k_tick_process;
    cl_kernel k_bfs_step;
    cl_kernel k_anomaly_detect;
    cl_kernel k_conservation_check;
    cl_kernel k_fleet_vibe_reduce;
    cl_kernel k_fleet_surprise_reduce;
    cl_kernel k_fleet_energy_reduce;

    /* Graph data */
    int room_count;
    int edge_count;
    int initialized;

    /* Device buffers */
    cl_mem d_vibes;             // [N x VIBE_DIM]
    cl_mem d_out_vibes;         // [N x VIBE_DIM] scratch for diffusion
    cl_mem d_neighbor_vibes;    // [M x VIBE_DIM]
    cl_mem d_neighbor_offsets;  // [N+1]
    cl_mem d_neighbor_indices;  // [M]
    cl_mem d_weights;           // [M]

    /* Perception / prediction */
    cl_mem d_perception_db;     // [N x max_window x D]
    cl_mem d_actual_db;         // [N x D]
    cl_mem d_window_sizes;      // [N]
    cl_mem d_predicted;         // [N x D]
    cl_mem d_surprise;          // [N]

    /* Murmur */
    cl_mem d_murmur_vibes;      // [MAX_MURMUR x D]
    cl_mem d_murmur_surprise;   // [MAX_MURMUR]
    cl_mem d_murmur_targets;    // [MAX_MURMUR]
    cl_mem d_murmur_ttl;        // [MAX_MURMUR]
    cl_mem d_murmur_ttl_out;    // [MAX_MURMUR]
    int murmur_count;

    /* Conservation */
    cl_mem d_perception_counts; // [N]
    cl_mem d_prediction_counts; // [N]
    cl_mem d_violations;        // [N]

    /* Fleet */
    cl_mem d_fleet_vibe;        // [D]
    cl_mem d_fleet_surprise;    // [1]
    cl_mem d_fleet_energy;      // [1]

    /* Energy */
    cl_mem d_energy;            // [N]

    /* BFS */
    cl_mem d_distances;         // [N]
    cl_mem d_frontier;          // [N]
    cl_mem d_next_frontier;     // [N]

    /* Anomaly */
    cl_mem d_anomalies;         // [N]

    /* Signal */
    cl_mem d_signal_src;        // [MAX_SIGNAL]
    cl_mem d_signal_dst;        // [MAX_SIGNAL]
    cl_mem d_signal_deadband;   // [MAX_SIGNAL]
    cl_mem d_signal_out;        // [MAX_SIGNAL x D]
    cl_mem d_signal_active;     // [MAX_SIGNAL]

} GrandPatternCL;

/**
 * Initialize OpenCL context, compile kernels.
 * Returns NULL on failure.
 */
GrandPatternCL* gpcl_init(void);

/**
 * Release all OpenCL resources.
 */
void gpcl_destroy(GrandPatternCL* cl);

/**
 * Allocate graph buffers for room_count rooms.
 * Returns 0 on success.
 */
int gpcl_create_graph(GrandPatternCL* cl, int room_count);

/**
 * Set graph edges (CSR format). from/to are room IDs, weights are edge weights.
 * Returns 0 on success.
 */
int gpcl_set_edges(GrandPatternCL* cl, int* from, int* to, double* weights, int edge_count);

/**
 * Set murmur data for gossip round.
 */
int gpcl_set_murmurs(GrandPatternCL* cl, double* vibes, double* surprise,
                      int* targets, int* ttl, int count);

/**
 * Full tick cycle: diffuse vibes → predict → tick process → fleet reduce.
 * Returns 0 on success.
 */
int gpcl_tick(GrandPatternCL* cl);

/**
 * Run gossip propagation round.
 */
int gpcl_gossip(GrandPatternCL* cl, double blend_rate);

/**
 * Diffuse vibes with given coefficient.
 */
int gpcl_diffuse_vibes(GrandPatternCL* cl, double coeff);

/**
 * Run JEPA prediction for all rooms.
 */
int gpcl_predict_all(GrandPatternCL* cl);

/**
 * Check conservation invariant.
 */
int gpcl_check_conservation(GrandPatternCL* cl, int tolerance);

/**
 * Query results.
 */
int gpcl_get_vibes(GrandPatternCL* cl, double* out_vibes);
int gpcl_get_surprise(GrandPatternCL* cl, double* out_surprise);
int gpcl_get_fleet_vibe(GrandPatternCL* cl, double* out_fleet);
int gpcl_get_violations(GrandPatternCL* cl, int* out_violations);
int gpcl_get_predicted(GrandPatternCL* cl, double* out_predicted);
int gpcl_get_energy(GrandPatternCL* cl, double* out_energy);
int gpcl_get_distances(GrandPatternCL* cl, int* out_distances);
int gpcl_get_anomalies(GrandPatternCL* cl, int* out_anomalies);

/**
 * Set room vibes (host → device).
 */
int gpcl_set_vibes(GrandPatternCL* cl, const double* vibes);

/**
 * Set perception/prediction counts for conservation.
 */
int gpcl_set_conservation_data(GrandPatternCL* cl, const int* perc, const int* pred);

/**
 * BFS from source node. Runs iteratively until convergence.
 * Results in gpcl_get_distances.
 */
int gpcl_bfs(GrandPatternCL* cl, int source);

/**
 * Detect anomalies given thresholds.
 */
int gpcl_detect_anomalies(GrandPatternCL* cl, double surprise_thresh, double divergence_thresh);

/**
 * Get last error string.
 */
const char* gpcl_error_string(int err);

#endif /* GRAND_PATTERN_CL_H */
