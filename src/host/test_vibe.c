//
// test_vibe.c — Test vibe compute kernel (OpenCL)
//

#include "opencl_context.h"
#include <math.h>
#include <stdio.h>

int test_vibe(opencl_ctx_t *ctx) {
    printf("=== Test: Vibe Compute (OpenCL) ===\n");
    const int n = 2, dim = 4;
    float dt = 1.0f;

    float h_e[n*dim] = {1,0,0,0, 3,4,0,0};
    float h_v[n*dim] = {0,1,0,0, 0,0,1,0};
    float h_vb[n*dim];

    cl_mem d_e = ocl_alloc_copy(ctx, h_e, n*dim*sizeof(float));
    cl_mem d_v = ocl_alloc_copy(ctx, h_v, n*dim*sizeof(float));
    cl_mem d_vb = ocl_alloc(ctx, n*dim*sizeof(float));

    cl_uint u_n = n, u_dim = dim;
    clSetKernelArg(ctx->vibe_compute, 0, sizeof(cl_mem), &d_e);
    clSetKernelArg(ctx->vibe_compute, 1, sizeof(cl_mem), &d_v);
    clSetKernelArg(ctx->vibe_compute, 2, sizeof(cl_mem), &d_vb);
    clSetKernelArg(ctx->vibe_compute, 3, sizeof(cl_uint), &u_n);
    clSetKernelArg(ctx->vibe_compute, 4, sizeof(cl_uint), &u_dim);
    clSetKernelArg(ctx->vibe_compute, 5, sizeof(float), &dt);

    size_t global = n * 256, local = 256;
    clEnqueueNDRangeKernel(ctx->queue, ctx->vibe_compute, 1, NULL, &global, &local, 0, NULL, NULL);
    ocl_finish(ctx);
    ocl_copy_to_host(ctx, h_vb, d_vb, n*dim*sizeof(float));

    int pass = 1;
    for (int r = 0; r < n; r++) {
        float norm = 0;
        for (int j = 0; j < dim; j++) norm += h_vb[r*dim+j]*h_vb[r*dim+j];
        if (fabsf(norm-1.0f)>0.01f) { printf("FAIL room %d norm=%f\n",r,norm); pass=0; }
    }

    ocl_free(d_e); ocl_free(d_v); ocl_free(d_vb);
    printf("Test Vibe Compute: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}
