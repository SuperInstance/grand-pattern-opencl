//
// test_merge.c — Test merge candidates kernel (OpenCL)
//

#include "opencl_context.h"
#include <stdio.h>

int test_merge(opencl_ctx_t *ctx) {
    printf("=== Test: Merge Candidates (OpenCL) ===\n");
    const int n = 4, dim = 4;
    float threshold = 0.9f;

    float h_e[n*dim] = {1,0,0,0, 1,0.01f,0,0, 0,1,0,0, 0,1,0.01f,0};
    float h_s[n] = {1,1,1,1};
    cl_uint h_c[n], h_count = 0;

    cl_mem d_e = ocl_alloc_copy(ctx, h_e, n*dim*sizeof(float));
    cl_mem d_s = ocl_alloc_copy(ctx, h_s, n*sizeof(float));
    cl_mem d_c = ocl_alloc(ctx, n*sizeof(cl_uint));
    cl_mem d_count = ocl_alloc_copy(ctx, &h_count, sizeof(cl_uint));

    cl_uint u_n = n, u_dim = dim;
    clSetKernelArg(ctx->merge_candidates, 0, sizeof(cl_mem), &d_e);
    clSetKernelArg(ctx->merge_candidates, 1, sizeof(cl_mem), &d_s);
    clSetKernelArg(ctx->merge_candidates, 2, sizeof(cl_mem), &d_c);
    clSetKernelArg(ctx->merge_candidates, 3, sizeof(cl_mem), &d_count);
    clSetKernelArg(ctx->merge_candidates, 4, sizeof(cl_uint), &u_n);
    clSetKernelArg(ctx->merge_candidates, 5, sizeof(cl_uint), &u_dim);
    clSetKernelArg(ctx->merge_candidates, 6, sizeof(float), &threshold);

    size_t global = n, local = n;
    clEnqueueNDRangeKernel(ctx->queue, ctx->merge_candidates, 1, NULL, &global, &local, 0, NULL, NULL);
    ocl_finish(ctx);
    ocl_copy_to_host(ctx, h_c, d_c, n*sizeof(cl_uint));
    ocl_copy_to_host(ctx, &h_count, d_count, sizeof(cl_uint));

    int pass = 1;
    if (!h_c[0]) { printf("FAIL pair 0-1\n"); pass=0; }
    if (h_c[1]) { printf("FAIL pair 1-2\n"); pass=0; }
    if (!h_c[2]) { printf("FAIL pair 2-3\n"); pass=0; }

    ocl_free(d_e); ocl_free(d_s); ocl_free(d_c); ocl_free(d_count);
    printf("Test Merge Candidates: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}
