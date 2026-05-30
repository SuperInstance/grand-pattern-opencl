//
// test_correlation.c — Test correlation matrix kernel (OpenCL)
//

#include "opencl_context.h"
#include <math.h>
#include <stdio.h>

int test_correlation(opencl_ctx_t *ctx) {
    printf("=== Test: Correlation Matrix (OpenCL) ===\n");
    const int nr = 3, dim = 4;

    float h_v[nr*dim] = {1,0,0,0, 0,1,0,0, 1,0,0,0};
    float h_m[nr*nr];

    cl_mem d_v = ocl_alloc_copy(ctx, h_v, nr*dim*sizeof(float));
    cl_mem d_m = ocl_alloc(ctx, nr*nr*sizeof(float));

    cl_uint u_nr = nr, u_dim = dim;
    clSetKernelArg(ctx->correlation_matrix, 0, sizeof(cl_mem), &d_v);
    clSetKernelArg(ctx->correlation_matrix, 1, sizeof(cl_mem), &d_m);
    clSetKernelArg(ctx->correlation_matrix, 2, sizeof(cl_uint), &u_nr);
    clSetKernelArg(ctx->correlation_matrix, 3, sizeof(cl_uint), &u_dim);

    size_t global[2] = {nr, nr}, local[2] = {1, 1};
    clEnqueueNDRangeKernel(ctx->queue, ctx->correlation_matrix, 2, NULL, global, local, 0, NULL, NULL);
    ocl_finish(ctx);
    ocl_copy_to_host(ctx, h_m, d_m, nr*nr*sizeof(float));

    int pass = 1;
    for (int i = 0; i < nr; i++) {
        if (fabsf(h_m[i*nr+i]-1.0f)>0.01f) { printf("FAIL diag[%d]=%f\n",i,h_m[i*nr+i]); pass=0; }
    }
    if (fabsf(h_m[1])>0.01f) { printf("FAIL sim01=%f\n",h_m[1]); pass=0; }
    if (fabsf(h_m[2]-1.0f)>0.01f) { printf("FAIL sim02=%f\n",h_m[2]); pass=0; }

    ocl_free(d_v); ocl_free(d_m);
    printf("Test Correlation Matrix: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}
