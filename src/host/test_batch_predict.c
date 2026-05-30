//
// test_batch_predict.c — Test batch prediction kernel (OpenCL)
//

#include "opencl_context.h"
#include <math.h>
#include <stdio.h>

int test_batch_predict(opencl_ctx_t *ctx) {
    printf("=== Test: Batch Predict (OpenCL) ===\n");
    const int n = 3, dim = 4;
    float delta = 0.5f;

    float h_p[n*dim], h_t[n*dim], h_r[n*dim];
    for (int i = 0; i < n*dim; i++) { h_p[i] = 1.0f; h_t[i] = 3.0f; }

    cl_mem d_p = ocl_alloc_copy(ctx, h_p, n*dim*sizeof(float));
    cl_mem d_t = ocl_alloc_copy(ctx, h_t, n*dim*sizeof(float));
    cl_mem d_r = ocl_alloc(ctx, n*dim*sizeof(float));

    cl_uint u_n = n, u_dim = dim;
    clSetKernelArg(ctx->batch_predict, 0, sizeof(cl_mem), &d_p);
    clSetKernelArg(ctx->batch_predict, 1, sizeof(cl_mem), &d_t);
    clSetKernelArg(ctx->batch_predict, 2, sizeof(cl_mem), &d_r);
    clSetKernelArg(ctx->batch_predict, 3, sizeof(cl_uint), &u_n);
    clSetKernelArg(ctx->batch_predict, 4, sizeof(cl_uint), &u_dim);
    clSetKernelArg(ctx->batch_predict, 5, sizeof(float), &delta);

    size_t global = n * 256, local = 256;
    clEnqueueNDRangeKernel(ctx->queue, ctx->batch_predict, 1, NULL, &global, &local, 0, NULL, NULL);
    ocl_finish(ctx);
    ocl_copy_to_host(ctx, h_r, d_r, n*dim*sizeof(float));

    int pass = 1;
    for (int i = 0; i < n*dim; i++) {
        if (fabsf(h_r[i]-2.0f)>0.01f) { printf("FAIL [%d]=%f\n",i,h_r[i]); pass=0; }
    }

    ocl_free(d_p); ocl_free(d_t); ocl_free(d_r);
    printf("Test Batch Predict: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}
