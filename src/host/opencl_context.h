//
// opencl_context.h — OpenCL context and kernel management
//

#ifndef OPENCL_CONTEXT_H
#define OPENCL_CONTEXT_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    cl_kernel cosine_similarity;
    cl_kernel batch_predict;
    cl_kernel balance_check;
    cl_kernel decay;
    cl_kernel vibe_compute;
    cl_kernel correlation_matrix;
    cl_kernel merge_candidates;
} opencl_ctx_t;

int ocl_init(const char *kernel_dir, opencl_ctx_t *ctx);
void ocl_cleanup(opencl_ctx_t *ctx);

cl_mem ocl_alloc(opencl_ctx_t *ctx, size_t size);
cl_mem ocl_alloc_copy(opencl_ctx_t *ctx, const void *data, size_t size);
int ocl_copy_to_host(opencl_ctx_t *ctx, void *host, cl_mem device, size_t size);
void ocl_free(cl_mem mem);
int ocl_finish(opencl_ctx_t *ctx);

#endif
