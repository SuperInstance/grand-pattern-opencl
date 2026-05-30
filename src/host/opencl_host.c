/**
 * opencl_host.c — Implementation of Grand Pattern OpenCL host API.
 */

#include "opencl_host.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char last_error[512] = "";

static void set_error(const char* msg) {
    snprintf(last_error, sizeof(last_error), "%s", msg);
}

const char* gpcl_error_string(int err) {
    (void)err;
    return last_error;
}

static int check_cl(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        snprintf(last_error, sizeof(last_error), "%s (cl_int=%d)", msg, (int)err);
        return -1;
    }
    return 0;
}

/* Concatenate all kernel source files into one string */
static char* load_kernel_source(void) {
    /* We embed the kernel sources as string literals for portability.
       In production, you'd load from files. Here we reference the .cl files. */
    const char* kernel_files[] = {
        "src/kernels/vibe_diffusion.cl",
        "src/kernels/jepa_predict.cl",
        "src/kernels/murmur_gossip.cl",
        "src/kernels/signal_route.cl",
        "src/kernels/tick_process.cl",
        "src/kernels/graph_ops.cl",
        "src/kernels/conservation.cl",
        "src/kernels/fleet_ops.cl",
        NULL
    };

    /* Calculate total size */
    size_t total = 1;
    int count = 0;
    for (int i = 0; kernel_files[i]; i++) {
        FILE* f = fopen(kernel_files[i], "rb");
        if (!f) {
            char buf[256];
            snprintf(buf, sizeof(buf), "Cannot open kernel file: %s", kernel_files[i]);
            set_error(buf);
            return NULL;
        }
        fseek(f, 0, SEEK_END);
        total += (size_t)ftell(f) + 2; /* +2 for newlines */
        fclose(f);
        count++;
    }

    char* source = (char*)malloc(total);
    if (!source) { set_error("malloc failed"); return NULL; }
    source[0] = '\0';

    for (int i = 0; kernel_files[i]; i++) {
        FILE* f = fopen(kernel_files[i], "rb");
        if (!f) { free(source); return NULL; }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        size_t cur = strlen(source);
        fread(source + cur, 1, (size_t)sz, f);
        source[cur + sz] = '\n';
        source[cur + sz + 1] = '\0';
        fclose(f);
    }

    return source;
}

static cl_kernel get_kernel(GrandPatternCL* cl, const char* name) {
    cl_int err;
    cl_kernel k = clCreateKernel(cl->program, name, &err);
    if (err != CL_SUCCESS) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Failed to create kernel: %s (err=%d)", name, (int)err);
        set_error(buf);
        return NULL;
    }
    return k;
}

GrandPatternCL* gpcl_init(void) {
    GrandPatternCL* cl = (GrandPatternCL*)calloc(1, sizeof(GrandPatternCL));
    if (!cl) { set_error("calloc failed"); return NULL; }

    cl_int err;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &cl->platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        set_error("No OpenCL platform found");
        free(cl);
        return NULL;
    }

    err = clGetDeviceIDs(cl->platform, CL_DEVICE_TYPE_ALL, 1, &cl->device, NULL);
    if (err != CL_SUCCESS) {
        /* Try CPU fallback */
        err = clGetDeviceIDs(cl->platform, CL_DEVICE_TYPE_CPU, 1, &cl->device, NULL);
        if (err != CL_SUCCESS) {
            set_error("No OpenCL device found");
            free(cl);
            return NULL;
        }
    }

    cl->context = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &err);
    if (check_cl(err, "clCreateContext")) { free(cl); return NULL; }

#ifdef CL_VERSION_2_0
    cl->queue = clCreateCommandQueueWithProperties(cl->context, cl->device, NULL, &err);
#else
    cl->queue = clCreateCommandQueue(cl->context, cl->device, 0, &err);
#endif
    if (check_cl(err, "clCreateCommandQueue")) {
        clReleaseContext(cl->context);
        free(cl);
        return NULL;
    }

    /* Load and compile kernels */
    char* source = load_kernel_source();
    if (!source) {
        clReleaseCommandQueue(cl->queue);
        clReleaseContext(cl->context);
        free(cl);
        return NULL;
    }

    const char* src_ptr = source;
    cl->program = clCreateProgramWithSource(cl->context, 1, &src_ptr, NULL, &err);
    free(source);
    if (check_cl(err, "clCreateProgramWithSource")) {
        clReleaseCommandQueue(cl->queue);
        clReleaseContext(cl->context);
        free(cl);
        return NULL;
    }

    err = clBuildProgram(cl->program, 1, &cl->device,
                         "-cl-mad-enable -cl-no-signed-zeros -cl-denorms-are-zero",
                         NULL, NULL);
    if (err != CL_SUCCESS) {
        char build_log[4096];
        clGetProgramBuildInfo(cl->program, cl->device, CL_PROGRAM_BUILD_LOG,
                              sizeof(build_log), build_log, NULL);
        fprintf(stderr, "OpenCL build error:\n%s\n", build_log);
        set_error("Kernel build failed");
        clReleaseProgram(cl->program);
        clReleaseCommandQueue(cl->queue);
        clReleaseContext(cl->context);
        free(cl);
        return NULL;
    }

    /* Create kernels */
    cl->k_vibe_diffuse        = get_kernel(cl, "vibe_diffuse");
    cl->k_jepa_predict        = get_kernel(cl, "jepa_predict");
    cl->k_murmur_propagate    = get_kernel(cl, "murmur_propagate");
    cl->k_signal_route        = get_kernel(cl, "signal_route");
    cl->k_tick_process        = get_kernel(cl, "tick_process");
    cl->k_bfs_step            = get_kernel(cl, "bfs_step");
    cl->k_anomaly_detect      = get_kernel(cl, "anomaly_detect");
    cl->k_conservation_check  = get_kernel(cl, "conservation_check");
    cl->k_fleet_vibe_reduce   = get_kernel(cl, "fleet_vibe_reduce");
    cl->k_fleet_surprise_reduce = get_kernel(cl, "fleet_surprise_reduce");
    cl->k_fleet_energy_reduce = get_kernel(cl, "fleet_energy_reduce");

    if (!cl->k_vibe_diffuse || !cl->k_jepa_predict || !cl->k_murmur_propagate ||
        !cl->k_signal_route || !cl->k_tick_process || !cl->k_bfs_step ||
        !cl->k_anomaly_detect || !cl->k_conservation_check ||
        !cl->k_fleet_vibe_reduce || !cl->k_fleet_surprise_reduce ||
        !cl->k_fleet_energy_reduce) {
        gpcl_destroy(cl);
        return NULL;
    }

    cl->initialized = 1;
    return cl;
}

void gpcl_destroy(GrandPatternCL* cl) {
    if (!cl) return;

    /* Release kernels */
    if (cl->k_vibe_diffuse)          clReleaseKernel(cl->k_vibe_diffuse);
    if (cl->k_jepa_predict)          clReleaseKernel(cl->k_jepa_predict);
    if (cl->k_murmur_propagate)      clReleaseKernel(cl->k_murmur_propagate);
    if (cl->k_signal_route)          clReleaseKernel(cl->k_signal_route);
    if (cl->k_tick_process)          clReleaseKernel(cl->k_tick_process);
    if (cl->k_bfs_step)              clReleaseKernel(cl->k_bfs_step);
    if (cl->k_anomaly_detect)        clReleaseKernel(cl->k_anomaly_detect);
    if (cl->k_conservation_check)    clReleaseKernel(cl->k_conservation_check);
    if (cl->k_fleet_vibe_reduce)     clReleaseKernel(cl->k_fleet_vibe_reduce);
    if (cl->k_fleet_surprise_reduce) clReleaseKernel(cl->k_fleet_surprise_reduce);
    if (cl->k_fleet_energy_reduce)   clReleaseKernel(cl->k_fleet_energy_reduce);

    /* Release buffers */
    if (cl->d_vibes)             clReleaseMemObject(cl->d_vibes);
    if (cl->d_out_vibes)         clReleaseMemObject(cl->d_out_vibes);
    if (cl->d_neighbor_vibes)    clReleaseMemObject(cl->d_neighbor_vibes);
    if (cl->d_neighbor_offsets)  clReleaseMemObject(cl->d_neighbor_offsets);
    if (cl->d_neighbor_indices)  clReleaseMemObject(cl->d_neighbor_indices);
    if (cl->d_weights)           clReleaseMemObject(cl->d_weights);
    if (cl->d_perception_db)     clReleaseMemObject(cl->d_perception_db);
    if (cl->d_actual_db)         clReleaseMemObject(cl->d_actual_db);
    if (cl->d_window_sizes)      clReleaseMemObject(cl->d_window_sizes);
    if (cl->d_predicted)         clReleaseMemObject(cl->d_predicted);
    if (cl->d_surprise)          clReleaseMemObject(cl->d_surprise);
    if (cl->d_murmur_vibes)      clReleaseMemObject(cl->d_murmur_vibes);
    if (cl->d_murmur_surprise)   clReleaseMemObject(cl->d_murmur_surprise);
    if (cl->d_murmur_targets)    clReleaseMemObject(cl->d_murmur_targets);
    if (cl->d_murmur_ttl)        clReleaseMemObject(cl->d_murmur_ttl);
    if (cl->d_murmur_ttl_out)    clReleaseMemObject(cl->d_murmur_ttl_out);
    if (cl->d_perception_counts) clReleaseMemObject(cl->d_perception_counts);
    if (cl->d_prediction_counts) clReleaseMemObject(cl->d_prediction_counts);
    if (cl->d_violations)        clReleaseMemObject(cl->d_violations);
    if (cl->d_fleet_vibe)        clReleaseMemObject(cl->d_fleet_vibe);
    if (cl->d_fleet_surprise)    clReleaseMemObject(cl->d_fleet_surprise);
    if (cl->d_fleet_energy)      clReleaseMemObject(cl->d_fleet_energy);
    if (cl->d_energy)            clReleaseMemObject(cl->d_energy);
    if (cl->d_distances)         clReleaseMemObject(cl->d_distances);
    if (cl->d_frontier)          clReleaseMemObject(cl->d_frontier);
    if (cl->d_next_frontier)     clReleaseMemObject(cl->d_next_frontier);
    if (cl->d_anomalies)         clReleaseMemObject(cl->d_anomalies);
    if (cl->d_signal_src)        clReleaseMemObject(cl->d_signal_src);
    if (cl->d_signal_dst)        clReleaseMemObject(cl->d_signal_dst);
    if (cl->d_signal_deadband)   clReleaseMemObject(cl->d_signal_deadband);
    if (cl->d_signal_out)        clReleaseMemObject(cl->d_signal_out);
    if (cl->d_signal_active)     clReleaseMemObject(cl->d_signal_active);

    /* Release OpenCL objects */
    if (cl->program) clReleaseProgram(cl->program);
    if (cl->queue)   clReleaseCommandQueue(cl->queue);
    if (cl->context) clReleaseContext(cl->context);

    free(cl);
}

static cl_mem create_buffer(GrandPatternCL* cl, size_t size, cl_mem_flags flags) {
    cl_int err;
    cl_mem buf = clCreateBuffer(cl->context, flags, size, NULL, &err);
    if (err != CL_SUCCESS) {
        set_error("clCreateBuffer failed");
        return NULL;
    }
    return buf;
}

int gpcl_create_graph(GrandPatternCL* cl, int room_count) {
    if (!cl || !cl->initialized) { set_error("Not initialized"); return -1; }
    if (room_count <= 0) { set_error("Invalid room_count"); return -1; }

    cl->room_count = room_count;
    int N = room_count;
    int D = GP_VIBE_DIM;

    /* Allocate all buffers */
    cl->d_vibes            = create_buffer(cl, N * D * sizeof(double),
                                           CL_MEM_READ_WRITE);
    cl->d_out_vibes        = create_buffer(cl, N * D * sizeof(double),
                                           CL_MEM_READ_WRITE);
    cl->d_perception_db    = create_buffer(cl, N * GP_MAX_WINDOW * D * sizeof(double),
                                           CL_MEM_READ_WRITE);
    cl->d_actual_db        = create_buffer(cl, N * D * sizeof(double),
                                           CL_MEM_READ_WRITE);
    cl->d_window_sizes     = create_buffer(cl, N * sizeof(int),
                                           CL_MEM_READ_WRITE);
    cl->d_predicted        = create_buffer(cl, N * D * sizeof(double),
                                           CL_MEM_READ_WRITE);
    cl->d_surprise         = create_buffer(cl, N * sizeof(double),
                                           CL_MEM_READ_WRITE);
    cl->d_perception_counts = create_buffer(cl, N * sizeof(int),
                                            CL_MEM_READ_WRITE);
    cl->d_prediction_counts = create_buffer(cl, N * sizeof(int),
                                            CL_MEM_READ_WRITE);
    cl->d_violations       = create_buffer(cl, N * sizeof(int),
                                           CL_MEM_READ_WRITE);
    cl->d_fleet_vibe       = create_buffer(cl, D * sizeof(double),
                                           CL_MEM_READ_WRITE);
    cl->d_fleet_surprise   = create_buffer(cl, sizeof(double),
                                           CL_MEM_READ_WRITE);
    cl->d_fleet_energy     = create_buffer(cl, sizeof(double),
                                           CL_MEM_READ_WRITE);
    cl->d_energy           = create_buffer(cl, N * sizeof(double),
                                           CL_MEM_READ_WRITE);
    cl->d_distances        = create_buffer(cl, N * sizeof(int),
                                           CL_MEM_READ_WRITE);
    cl->d_frontier         = create_buffer(cl, N * sizeof(int),
                                           CL_MEM_READ_WRITE);
    cl->d_next_frontier    = create_buffer(cl, N * sizeof(int),
                                           CL_MEM_READ_WRITE);
    cl->d_anomalies        = create_buffer(cl, N * sizeof(int),
                                           CL_MEM_READ_WRITE);

    /* Check all allocations */
    if (!cl->d_vibes || !cl->d_out_vibes || !cl->d_perception_db ||
        !cl->d_actual_db || !cl->d_window_sizes || !cl->d_predicted ||
        !cl->d_surprise || !cl->d_perception_counts || !cl->d_prediction_counts ||
        !cl->d_violations || !cl->d_fleet_vibe || !cl->d_fleet_surprise ||
        !cl->d_fleet_energy || !cl->d_energy || !cl->d_distances ||
        !cl->d_frontier || !cl->d_next_frontier || !cl->d_anomalies) {
        return -1;
    }

    /* Initialize window sizes and energy to zero */
    int* zeros_i = (int*)calloc(N, sizeof(int));
    double* zeros_d = (double*)calloc(N, sizeof(double));
    clEnqueueWriteBuffer(cl->queue, cl->d_window_sizes, CL_TRUE, 0,
                         N * sizeof(int), zeros_i, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl->queue, cl->d_energy, CL_TRUE, 0,
                         N * sizeof(double), zeros_d, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl->queue, cl->d_perception_counts, CL_TRUE, 0,
                         N * sizeof(int), zeros_i, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl->queue, cl->d_prediction_counts, CL_TRUE, 0,
                         N * sizeof(int), zeros_i, 0, NULL, NULL);
    free(zeros_i);
    free(zeros_d);

    return 0;
}

int gpcl_set_edges(GrandPatternCL* cl, int* from, int* to, double* weights, int edge_count) {
    if (!cl || !cl->initialized) return -1;
    cl->edge_count = edge_count;
    int N = cl->room_count;
    int M = edge_count;

    /* Build CSR format */
    int* offsets = (int*)calloc(N + 1, sizeof(int));
    int* indices = (int*)malloc(M * sizeof(int));
    double* w    = (double*)malloc(M * sizeof(double));
    double* nv   = (double*)calloc(M * GP_VIBE_DIM, sizeof(double));

    if (!offsets || !indices || !w || !nv) {
        free(offsets); free(indices); free(w); free(nv);
        return -1;
    }

    /* Count edges per room */
    for (int e = 0; e < M; e++) offsets[from[e] + 1]++;
    /* Prefix sum */
    for (int i = 1; i <= N; i++) offsets[i] += offsets[i - 1];

    /* Fill indices and weights (use temp counter) */
    int* temp = (int*)calloc(N, sizeof(int));
    for (int e = 0; e < M; e++) {
        int room = from[e];
        int pos = offsets[room] + temp[room];
        indices[pos] = to[e];
        w[pos] = weights[e];
        temp[room]++;
    }
    free(temp);

    /* Allocate device buffers */
    cl->d_neighbor_offsets = create_buffer(cl, (N + 1) * sizeof(int),
                                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    cl->d_neighbor_indices = create_buffer(cl, M * sizeof(int),
                                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    cl->d_weights          = create_buffer(cl, M * sizeof(double),
                                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    cl->d_neighbor_vibes   = create_buffer(cl, M * GP_VIBE_DIM * sizeof(double),
                                           CL_MEM_READ_WRITE);

    if (!cl->d_neighbor_offsets || !cl->d_neighbor_indices ||
        !cl->d_weights || !cl->d_neighbor_vibes) {
        free(offsets); free(indices); free(w); free(nv);
        return -1;
    }

    /* Upload CSR data */
    clEnqueueWriteBuffer(cl->queue, cl->d_neighbor_offsets, CL_TRUE, 0,
                         (N + 1) * sizeof(int), offsets, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl->queue, cl->d_neighbor_indices, CL_TRUE, 0,
                         M * sizeof(int), indices, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl->queue, cl->d_weights, CL_TRUE, 0,
                         M * sizeof(double), w, 0, NULL, NULL);

    /* Upload initial zero neighbor vibes */
    clEnqueueWriteBuffer(cl->queue, cl->d_neighbor_vibes, CL_TRUE, 0,
                         M * GP_VIBE_DIM * sizeof(double), nv, 0, NULL, NULL);

    free(offsets); free(indices); free(w); free(nv);
    return 0;
}

int gpcl_set_vibes(GrandPatternCL* cl, const double* vibes) {
    if (!cl || !cl->d_vibes) return -1;
    size_t sz = (size_t)cl->room_count * GP_VIBE_DIM * sizeof(double);
    cl_int err = clEnqueueWriteBuffer(cl->queue, cl->d_vibes, CL_TRUE, 0,
                                      sz, vibes, 0, NULL, NULL);
    return check_cl(err, "gpcl_set_vibes write");
}

int gpcl_set_conservation_data(GrandPatternCL* cl, const int* perc, const int* pred) {
    if (!cl) return -1;
    size_t sz = (size_t)cl->room_count * sizeof(int);
    clEnqueueWriteBuffer(cl->queue, cl->d_perception_counts, CL_TRUE, 0, sz, perc, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl->queue, cl->d_prediction_counts, CL_TRUE, 0, sz, pred, 0, NULL, NULL);
    return 0;
}

int gpcl_set_murmurs(GrandPatternCL* cl, double* vibes, double* surprise,
                      int* targets, int* ttl, int count) {
    if (!cl) return -1;
    cl->murmur_count = count;
    /* Allocate murmur buffers if needed */
    size_t vibe_sz = (size_t)count * GP_VIBE_DIM * sizeof(double);
    size_t s_sz = (size_t)count * sizeof(double);
    size_t i_sz = (size_t)count * sizeof(int);

    if (cl->d_murmur_vibes) clReleaseMemObject(cl->d_murmur_vibes);
    if (cl->d_murmur_surprise) clReleaseMemObject(cl->d_murmur_surprise);
    if (cl->d_murmur_targets) clReleaseMemObject(cl->d_murmur_targets);
    if (cl->d_murmur_ttl) clReleaseMemObject(cl->d_murmur_ttl);
    if (cl->d_murmur_ttl_out) clReleaseMemObject(cl->d_murmur_ttl_out);

    cl->d_murmur_vibes    = create_buffer(cl, vibe_sz, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    cl->d_murmur_surprise = create_buffer(cl, s_sz, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    cl->d_murmur_targets  = create_buffer(cl, i_sz, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    cl->d_murmur_ttl      = create_buffer(cl, i_sz, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    cl->d_murmur_ttl_out  = create_buffer(cl, i_sz, CL_MEM_READ_WRITE);

    clEnqueueWriteBuffer(cl->queue, cl->d_murmur_vibes, CL_TRUE, 0, vibe_sz, vibes, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl->queue, cl->d_murmur_surprise, CL_TRUE, 0, s_sz, surprise, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl->queue, cl->d_murmur_targets, CL_TRUE, 0, i_sz, targets, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl->queue, cl->d_murmur_ttl, CL_TRUE, 0, i_sz, ttl, 0, NULL, NULL);

    return 0;
}

/* Helper: update neighbor_vibes from current vibes using CSR indices */
static int update_neighbor_vibes(GrandPatternCL* cl) {
    /* Read current vibes, index into neighbor_vibes. Do on host for simplicity. */
    int N = cl->room_count;
    int M = cl->edge_count;
    double* room_v = (double*)malloc(N * GP_VIBE_DIM * sizeof(double));
    int* offsets = (int*)malloc((N + 1) * sizeof(int));
    int* indices = (int*)malloc(M * sizeof(int));
    double* nv = (double*)malloc(M * GP_VIBE_DIM * sizeof(double));

    if (!room_v || !offsets || !indices || !nv) {
        free(room_v); free(offsets); free(indices); free(nv);
        return -1;
    }

    clEnqueueReadBuffer(cl->queue, cl->d_vibes, CL_TRUE, 0,
                        N * GP_VIBE_DIM * sizeof(double), room_v, 0, NULL, NULL);
    clEnqueueReadBuffer(cl->queue, cl->d_neighbor_offsets, CL_TRUE, 0,
                        (N + 1) * sizeof(int), offsets, 0, NULL, NULL);
    clEnqueueReadBuffer(cl->queue, cl->d_neighbor_indices, CL_TRUE, 0,
                        M * sizeof(int), indices, 0, NULL, NULL);

    /* Build neighbor_vibes: for each edge e (offset-based), copy vibes of target */
    for (int r = 0; r < N; r++) {
        for (int e = offsets[r]; e < offsets[r + 1]; e++) {
            int nb = indices[e];
            memcpy(&nv[e * GP_VIBE_DIM], &room_v[nb * GP_VIBE_DIM],
                   GP_VIBE_DIM * sizeof(double));
        }
    }

    clEnqueueWriteBuffer(cl->queue, cl->d_neighbor_vibes, CL_TRUE, 0,
                         M * GP_VIBE_DIM * sizeof(double), nv, 0, NULL, NULL);

    free(room_v); free(offsets); free(indices); free(nv);
    return 0;
}

int gpcl_diffuse_vibes(GrandPatternCL* cl, double coeff) {
    if (!cl || !cl->k_vibe_diffuse) return -1;

    int rc = update_neighbor_vibes(cl);
    if (rc) return rc;

    /* Also copy vibes to actual_db for prediction */
    clEnqueueCopyBuffer(cl->queue, cl->d_vibes, cl->d_actual_db, 0, 0,
                        (size_t)cl->room_count * GP_VIBE_DIM * sizeof(double),
                        0, NULL, NULL);

    cl_int err;
    int arg = 0;
    err = clSetKernelArg(cl->k_vibe_diffuse, arg++, sizeof(cl_mem), &cl->d_vibes);
    err |= clSetKernelArg(cl->k_vibe_diffuse, arg++, sizeof(cl_mem), &cl->d_neighbor_vibes);
    err |= clSetKernelArg(cl->k_vibe_diffuse, arg++, sizeof(cl_mem), &cl->d_neighbor_offsets);
    err |= clSetKernelArg(cl->k_vibe_diffuse, arg++, sizeof(cl_mem), &cl->d_weights);
    err |= clSetKernelArg(cl->k_vibe_diffuse, arg++, sizeof(double), &coeff);
    err |= clSetKernelArg(cl->k_vibe_diffuse, arg++, sizeof(cl_mem), &cl->d_out_vibes);
    if (check_cl(err, "vibe_diffuse set args")) return -1;

    size_t global = (size_t)cl->room_count;
    err = clEnqueueNDRangeKernel(cl->queue, cl->k_vibe_diffuse, 1, NULL,
                                 &global, NULL, 0, NULL, NULL);
    if (check_cl(err, "vibe_diffuse enqueue")) return -1;

    /* Copy out_vibes back to vibes */
    clEnqueueCopyBuffer(cl->queue, cl->d_out_vibes, cl->d_vibes, 0, 0,
                        (size_t)cl->room_count * GP_VIBE_DIM * sizeof(double),
                        0, NULL, NULL);
    clFinish(cl->queue);
    return 0;
}

int gpcl_predict_all(GrandPatternCL* cl) {
    if (!cl || !cl->k_jepa_predict) return -1;

    cl_int err;
    int arg = 0;
    int max_window = GP_MAX_WINDOW;
    int dim = GP_VIBE_DIM;
    err = clSetKernelArg(cl->k_jepa_predict, arg++, sizeof(cl_mem), &cl->d_perception_db);
    err |= clSetKernelArg(cl->k_jepa_predict, arg++, sizeof(cl_mem), &cl->d_actual_db);
    err |= clSetKernelArg(cl->k_jepa_predict, arg++, sizeof(cl_mem), &cl->d_window_sizes);
    err |= clSetKernelArg(cl->k_jepa_predict, arg++, sizeof(int), &max_window);
    err |= clSetKernelArg(cl->k_jepa_predict, arg++, sizeof(int), &dim);
    err |= clSetKernelArg(cl->k_jepa_predict, arg++, sizeof(cl_mem), &cl->d_predicted);
    err |= clSetKernelArg(cl->k_jepa_predict, arg++, sizeof(cl_mem), &cl->d_surprise);
    if (check_cl(err, "jepa_predict set args")) return -1;

    size_t global = (size_t)cl->room_count;
    err = clEnqueueNDRangeKernel(cl->queue, cl->k_jepa_predict, 1, NULL,
                                 &global, NULL, 0, NULL, NULL);
    if (check_cl(err, "jepa_predict enqueue")) return -1;

    clFinish(cl->queue);
    return 0;
}

int gpcl_gossip(GrandPatternCL* cl, double blend_rate) {
    if (!cl || !cl->k_murmur_propagate || cl->murmur_count <= 0) return -1;

    cl_int err;
    int arg = 0;
    err = clSetKernelArg(cl->k_murmur_propagate, arg++, sizeof(cl_mem), &cl->d_murmur_vibes);
    err |= clSetKernelArg(cl->k_murmur_propagate, arg++, sizeof(cl_mem), &cl->d_murmur_surprise);
    err |= clSetKernelArg(cl->k_murmur_propagate, arg++, sizeof(cl_mem), &cl->d_murmur_targets);
    err |= clSetKernelArg(cl->k_murmur_propagate, arg++, sizeof(cl_mem), &cl->d_murmur_ttl);
    err |= clSetKernelArg(cl->k_murmur_propagate, arg++, sizeof(double), &blend_rate);
    err |= clSetKernelArg(cl->k_murmur_propagate, arg++, sizeof(cl_mem), &cl->d_vibes);
    err |= clSetKernelArg(cl->k_murmur_propagate, arg++, sizeof(cl_mem), &cl->d_surprise);
    err |= clSetKernelArg(cl->k_murmur_propagate, arg++, sizeof(cl_mem), &cl->d_murmur_ttl_out);
    if (check_cl(err, "murmur_propagate set args")) return -1;

    size_t global = (size_t)cl->murmur_count;
    err = clEnqueueNDRangeKernel(cl->queue, cl->k_murmur_propagate, 1, NULL,
                                 &global, NULL, 0, NULL, NULL);
    if (check_cl(err, "murmur_propagate enqueue")) return -1;

    clFinish(cl->queue);
    return 0;
}

int gpcl_check_conservation(GrandPatternCL* cl, int tolerance) {
    if (!cl || !cl->k_conservation_check) return -1;

    cl_int err;
    int arg = 0;
    int n = cl->room_count;
    err = clSetKernelArg(cl->k_conservation_check, arg++, sizeof(cl_mem), &cl->d_perception_counts);
    err |= clSetKernelArg(cl->k_conservation_check, arg++, sizeof(cl_mem), &cl->d_prediction_counts);
    err |= clSetKernelArg(cl->k_conservation_check, arg++, sizeof(int), &tolerance);
    err |= clSetKernelArg(cl->k_conservation_check, arg++, sizeof(int), &n);
    err |= clSetKernelArg(cl->k_conservation_check, arg++, sizeof(cl_mem), &cl->d_violations);
    if (check_cl(err, "conservation_check set args")) return -1;

    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(cl->queue, cl->k_conservation_check, 1, NULL,
                                 &global, NULL, 0, NULL, NULL);
    if (check_cl(err, "conservation_check enqueue")) return -1;

    clFinish(cl->queue);
    return 0;
}

int gpcl_tick(GrandPatternCL* cl) {
    int rc;
    /* 1. Diffuse vibes */
    rc = gpcl_diffuse_vibes(cl, 0.1);
    if (rc) return rc;

    /* 2. Predict */
    rc = gpcl_predict_all(cl);
    if (rc) return rc;

    /* 3. Tick process: advance perception windows */
    if (cl->k_tick_process) {
        cl_int err;
        int arg = 0;
        int max_window = GP_MAX_WINDOW;
        int dim = GP_VIBE_DIM;
        double energy_decay = 0.95;
        err = clSetKernelArg(cl->k_tick_process, arg++, sizeof(cl_mem), &cl->d_perception_db);
        err |= clSetKernelArg(cl->k_tick_process, arg++, sizeof(cl_mem), &cl->d_window_sizes);
        err |= clSetKernelArg(cl->k_tick_process, arg++, sizeof(int), &max_window);
        err |= clSetKernelArg(cl->k_tick_process, arg++, sizeof(int), &dim);
        err |= clSetKernelArg(cl->k_tick_process, arg++, sizeof(cl_mem), &cl->d_vibes);
        err |= clSetKernelArg(cl->k_tick_process, arg++, sizeof(cl_mem), &cl->d_energy);
        err |= clSetKernelArg(cl->k_tick_process, arg++, sizeof(cl_mem), &cl->d_surprise);
        err |= clSetKernelArg(cl->k_tick_process, arg++, sizeof(double), &energy_decay);
        if (check_cl(err, "tick_process set args")) return -1;

        size_t global = (size_t)cl->room_count;
        err = clEnqueueNDRangeKernel(cl->queue, cl->k_tick_process, 1, NULL,
                                     &global, NULL, 0, NULL, NULL);
        if (check_cl(err, "tick_process enqueue")) return -1;
        clFinish(cl->queue);
    }

    /* 4. Fleet reduce */
    if (cl->k_fleet_vibe_reduce) {
        cl_int err;
        int rc2 = cl->room_count;
        err = clSetKernelArg(cl->k_fleet_vibe_reduce, 0, sizeof(cl_mem), &cl->d_vibes);
        err |= clSetKernelArg(cl->k_fleet_vibe_reduce, 1, sizeof(int), &rc2);
        err |= clSetKernelArg(cl->k_fleet_vibe_reduce, 2, sizeof(cl_mem), &cl->d_fleet_vibe);
        if (err == CL_SUCCESS) {
            size_t g = GP_VIBE_DIM;
            clEnqueueNDRangeKernel(cl->queue, cl->k_fleet_vibe_reduce, 1, NULL,
                                   &g, NULL, 0, NULL, NULL);
            clFinish(cl->queue);
        }
    }

    return 0;
}

int gpcl_bfs(GrandPatternCL* cl, int source) {
    if (!cl || !cl->k_bfs_step) return -1;
    int N = cl->room_count;

    /* Initialize distances: -1 everywhere, 0 at source */
    int* dist = (int*)malloc(N * sizeof(int));
    int* front = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) dist[i] = -1;
    dist[source] = 0;
    front[source] = 1;

    clEnqueueWriteBuffer(cl->queue, cl->d_distances, CL_TRUE, 0,
                         N * sizeof(int), dist, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl->queue, cl->d_frontier, CL_TRUE, 0,
                         N * sizeof(int), front, 0, NULL, NULL);

    free(dist); free(front);

    /* Run BFS steps until no more frontier */
    int max_iter = N; /* worst case */
    for (int iter = 0; iter < max_iter; iter++) {
        cl_int err;
        int n = N;
        err = clSetKernelArg(cl->k_bfs_step, 0, sizeof(cl_mem), &cl->d_neighbor_offsets);
        err |= clSetKernelArg(cl->k_bfs_step, 1, sizeof(cl_mem), &cl->d_neighbor_indices);
        err |= clSetKernelArg(cl->k_bfs_step, 2, sizeof(cl_mem), &cl->d_distances);
        err |= clSetKernelArg(cl->k_bfs_step, 3, sizeof(cl_mem), &cl->d_frontier);
        err |= clSetKernelArg(cl->k_bfs_step, 4, sizeof(cl_mem), &cl->d_next_frontier);
        err |= clSetKernelArg(cl->k_bfs_step, 5, sizeof(int), &n);
        if (check_cl(err, "bfs_step set args")) return -1;

        size_t global = (size_t)N;
        err = clEnqueueNDRangeKernel(cl->queue, cl->k_bfs_step, 1, NULL,
                                     &global, NULL, 0, NULL, NULL);
        clFinish(cl->queue);

        /* Check if next_frontier has any entries */
        int* nf = (int*)malloc(N * sizeof(int));
        clEnqueueReadBuffer(cl->queue, cl->d_next_frontier, CL_TRUE, 0,
                            N * sizeof(int), nf, 0, NULL, NULL);
        int any = 0;
        for (int i = 0; i < N; i++) { if (nf[i]) { any = 1; break; } }

        /* Swap frontier = next_frontier */
        clEnqueueCopyBuffer(cl->queue, cl->d_next_frontier, cl->d_frontier, 0, 0,
                            N * sizeof(int), 0, NULL, NULL);
        /* Clear next_frontier */
        int* zeros = (int*)calloc(N, sizeof(int));
        clEnqueueWriteBuffer(cl->queue, cl->d_next_frontier, CL_TRUE, 0,
                             N * sizeof(int), zeros, 0, NULL, NULL);
        free(nf); free(zeros);

        if (!any) break;
    }
    return 0;
}

int gpcl_detect_anomalies(GrandPatternCL* cl, double surprise_thresh, double divergence_thresh) {
    if (!cl || !cl->k_anomaly_detect) return -1;

    /* First compute fleet vibe */
    cl_int err;
    int rc = cl->room_count;
    err = clSetKernelArg(cl->k_fleet_vibe_reduce, 0, sizeof(cl_mem), &cl->d_vibes);
    err |= clSetKernelArg(cl->k_fleet_vibe_reduce, 1, sizeof(int), &rc);
    err |= clSetKernelArg(cl->k_fleet_vibe_reduce, 2, sizeof(cl_mem), &cl->d_fleet_vibe);
    if (!check_cl(err, "fleet reduce for anomaly")) {
        size_t g = GP_VIBE_DIM;
        clEnqueueNDRangeKernel(cl->queue, cl->k_fleet_vibe_reduce, 1, NULL,
                               &g, NULL, 0, NULL, NULL);
        clFinish(cl->queue);
    }

    int n = cl->room_count;
    err = clSetKernelArg(cl->k_anomaly_detect, 0, sizeof(cl_mem), &cl->d_vibes);
    err |= clSetKernelArg(cl->k_anomaly_detect, 1, sizeof(cl_mem), &cl->d_surprise);
    err |= clSetKernelArg(cl->k_anomaly_detect, 2, sizeof(cl_mem), &cl->d_fleet_vibe);
    err |= clSetKernelArg(cl->k_anomaly_detect, 3, sizeof(double), &surprise_thresh);
    err |= clSetKernelArg(cl->k_anomaly_detect, 4, sizeof(double), &divergence_thresh);
    err |= clSetKernelArg(cl->k_anomaly_detect, 5, sizeof(int), &n);
    err |= clSetKernelArg(cl->k_anomaly_detect, 6, sizeof(cl_mem), &cl->d_anomalies);
    if (check_cl(err, "anomaly_detect set args")) return -1;

    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(cl->queue, cl->k_anomaly_detect, 1, NULL,
                                 &global, NULL, 0, NULL, NULL);
    if (check_cl(err, "anomaly_detect enqueue")) return -1;
    clFinish(cl->queue);
    return 0;
}

/* Query functions */
int gpcl_get_vibes(GrandPatternCL* cl, double* out_vibes) {
    if (!cl || !cl->d_vibes) return -1;
    size_t sz = (size_t)cl->room_count * GP_VIBE_DIM * sizeof(double);
    cl_int err = clEnqueueReadBuffer(cl->queue, cl->d_vibes, CL_TRUE, 0,
                                     sz, out_vibes, 0, NULL, NULL);
    return check_cl(err, "get_vibes read");
}

int gpcl_get_surprise(GrandPatternCL* cl, double* out_surprise) {
    if (!cl) return -1;
    size_t sz = (size_t)cl->room_count * sizeof(double);
    cl_int err = clEnqueueReadBuffer(cl->queue, cl->d_surprise, CL_TRUE, 0,
                                     sz, out_surprise, 0, NULL, NULL);
    return check_cl(err, "get_surprise read");
}

int gpcl_get_fleet_vibe(GrandPatternCL* cl, double* out_fleet) {
    if (!cl) return -1;
    cl_int err = clEnqueueReadBuffer(cl->queue, cl->d_fleet_vibe, CL_TRUE, 0,
                                     GP_VIBE_DIM * sizeof(double), out_fleet,
                                     0, NULL, NULL);
    return check_cl(err, "get_fleet_vibe read");
}

int gpcl_get_violations(GrandPatternCL* cl, int* out_violations) {
    if (!cl) return -1;
    size_t sz = (size_t)cl->room_count * sizeof(int);
    cl_int err = clEnqueueReadBuffer(cl->queue, cl->d_violations, CL_TRUE, 0,
                                     sz, out_violations, 0, NULL, NULL);
    return check_cl(err, "get_violations read");
}

int gpcl_get_predicted(GrandPatternCL* cl, double* out_predicted) {
    if (!cl) return -1;
    size_t sz = (size_t)cl->room_count * GP_VIBE_DIM * sizeof(double);
    cl_int err = clEnqueueReadBuffer(cl->queue, cl->d_predicted, CL_TRUE, 0,
                                     sz, out_predicted, 0, NULL, NULL);
    return check_cl(err, "get_predicted read");
}

int gpcl_get_energy(GrandPatternCL* cl, double* out_energy) {
    if (!cl) return -1;
    size_t sz = (size_t)cl->room_count * sizeof(double);
    cl_int err = clEnqueueReadBuffer(cl->queue, cl->d_energy, CL_TRUE, 0,
                                     sz, out_energy, 0, NULL, NULL);
    return check_cl(err, "get_energy read");
}

int gpcl_get_distances(GrandPatternCL* cl, int* out_distances) {
    if (!cl) return -1;
    size_t sz = (size_t)cl->room_count * sizeof(int);
    cl_int err = clEnqueueReadBuffer(cl->queue, cl->d_distances, CL_TRUE, 0,
                                     sz, out_distances, 0, NULL, NULL);
    return check_cl(err, "get_distances read");
}

int gpcl_get_anomalies(GrandPatternCL* cl, int* out_anomalies) {
    if (!cl) return -1;
    size_t sz = (size_t)cl->room_count * sizeof(int);
    cl_int err = clEnqueueReadBuffer(cl->queue, cl->d_anomalies, CL_TRUE, 0,
                                     sz, out_anomalies, 0, NULL, NULL);
    return check_cl(err, "get_anomalies read");
}
