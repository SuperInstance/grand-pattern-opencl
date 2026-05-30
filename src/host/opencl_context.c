//
// opencl_context.c — OpenCL context and kernel management
//

#include "opencl_context.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CL(call) do { \
    cl_int err = (call); \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        return -1; \
    } \
} while(0)

static cl_kernel load_kernel(cl_program program, const char *name, cl_int *err) {
    cl_kernel k = clCreateKernel(program, name, err);
    return k;
}

static int load_kernels(opencl_ctx_t *ctx) {
    cl_int err;

    ctx->cosine_similarity = load_kernel(ctx->program, "cosine_similarity", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Missing kernel: cosine_similarity\n"); return -1; }

    ctx->batch_predict = load_kernel(ctx->program, "batch_predict", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Missing kernel: batch_predict\n"); return -1; }

    ctx->balance_check = load_kernel(ctx->program, "balance_check", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Missing kernel: balance_check\n"); return -1; }

    ctx->decay = load_kernel(ctx->program, "decay", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Missing kernel: decay\n"); return -1; }

    ctx->vibe_compute = load_kernel(ctx->program, "vibe_compute", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Missing kernel: vibe_compute\n"); return -1; }

    ctx->correlation_matrix = load_kernel(ctx->program, "correlation_matrix", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Missing kernel: correlation_matrix\n"); return -1; }

    ctx->merge_candidates = load_kernel(ctx->program, "merge_candidates", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Missing kernel: merge_candidates\n"); return -1; }

    return 0;
}

static char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(*size + 1);
    fread(buf, 1, *size, f);
    buf[*size] = '\0';
    fclose(f);
    return buf;
}

int ocl_init(const char *kernel_dir, opencl_ctx_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    cl_int err;

    // Get first platform
    CHECK_CL(clGetPlatformIDs(1, &ctx->platform, NULL));

    // Get first GPU device
    err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_GPU, 1, &ctx->device, NULL);
    if (err != CL_SUCCESS) {
        printf("No GPU found, trying any device...\n");
        CHECK_CL(clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_ALL, 1, &ctx->device, NULL));
    }

    ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);
    CHECK_CL(err);

    ctx->queue = clCreateCommandQueueWithProperties(ctx->context, ctx->device, 0, &err);
    CHECK_CL(err);

    // Load kernel source
    char path[512];
    snprintf(path, sizeof(path), "%s/kernels.cl", kernel_dir);
    size_t src_size;
    char *source = read_file(path, &src_size);
    if (!source) {
        fprintf(stderr, "Failed to read kernel file: %s\n", path);
        return -1;
    }

    const char *src_ptr = source;
    ctx->program = clCreateProgramWithSource(ctx->context, 1, &src_ptr, &src_size, &err);
    free(source);
    CHECK_CL(err);

    // Build with OpenCL 3.0 target (fallback to 2.0)
    err = clBuildProgram(ctx->program, 1, &ctx->device,
        "-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        // Try CL2.0
        err = clBuildProgram(ctx->program, 1, &ctx->device,
            "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
        if (err != CL_SUCCESS) {
            // Try without version
            err = clBuildProgram(ctx->program, 1, &ctx->device,
                "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
        }
    }
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size + 1);
        clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        return -1;
    }

    if (load_kernels(ctx) != 0) return -1;

    printf("OpenCL kernels loaded successfully\n");
    return 0;
}

void ocl_cleanup(opencl_ctx_t *ctx) {
    if (ctx->cosine_similarity) clReleaseKernel(ctx->cosine_similarity);
    if (ctx->batch_predict) clReleaseKernel(ctx->batch_predict);
    if (ctx->balance_check) clReleaseKernel(ctx->balance_check);
    if (ctx->decay) clReleaseKernel(ctx->decay);
    if (ctx->vibe_compute) clReleaseKernel(ctx->vibe_compute);
    if (ctx->correlation_matrix) clReleaseKernel(ctx->correlation_matrix);
    if (ctx->merge_candidates) clReleaseKernel(ctx->merge_candidates);
    if (ctx->program) clReleaseProgram(ctx->program);
    if (ctx->queue) clReleaseCommandQueue(ctx->queue);
    if (ctx->context) clReleaseContext(ctx->context);
}

cl_mem ocl_alloc(opencl_ctx_t *ctx, size_t size) {
    cl_int err;
    cl_mem mem = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS) return NULL;
    return mem;
}

cl_mem ocl_alloc_copy(opencl_ctx_t *ctx, const void *data, size_t size) {
    cl_int err;
    cl_mem mem = clCreateBuffer(ctx->context, CL_MEM_COPY_HOST_PTR, size, (void *)data, &err);
    if (err != CL_SUCCESS) return NULL;
    return mem;
}

int ocl_copy_to_host(opencl_ctx_t *ctx, void *host, cl_mem device, size_t size) {
    CHECK_CL(clEnqueueReadBuffer(ctx->queue, device, CL_TRUE, 0, size, host, 0, NULL, NULL));
    return 0;
}

void ocl_free(cl_mem mem) {
    if (mem) clReleaseMemObject(mem);
}

int ocl_finish(opencl_ctx_t *ctx) {
    CHECK_CL(clFinish(ctx->queue));
    return 0;
}
