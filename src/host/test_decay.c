//
// test_decay.c — Test decay kernel (OpenCL)
//

#include "opencl_context.h"
#include <math.h>
#include <stdio.h>

int test_decay(opencl_ctx_t *ctx) {
    printf("=== Test: Decay (OpenCL) ===\n");
    const int n = 8;

    float h_s[n], h_a[n];
    for (int i = 0; i < n; i++) { h_s[i] = 1.0f; h_a[i] = (float)i; }

    cl_mem d_s = ocl_alloc_copy(ctx, h_s, n*sizeof(float));
    cl_mem d_a = ocl_alloc_copy(ctx, h_a, n*sizeof(float));

    cl_uint u_n = n;
    float rate = 0.1f;
    clSetKernelArg(ctx->decay, 0, sizeof(cl_mem), &d_s);
    clSetKernelArg(ctx->decay, 1, sizeof(cl_mem), &d_a);
    clSetKernelArg(ctx->decay, 2, sizeof(cl_uint), &u_n);
    clSetKernelArg(ctx->decay, 3, sizeof(float), &rate);

    size_t global = n, local = n;
    clEnqueueNDRangeKernel(ctx->queue, ctx->decay, 1, NULL, &global, &local, 0, NULL, NULL);
    ocl_finish(ctx);
    ocl_copy_to_host(ctx, h_s, d_s, n*sizeof(float));

    int pass = 1;
    for (int i = 0; i < n; i++) {
        float expected = expf(-0.1f*(float)i);
        if (fabsf(h_s[i]-expected)>0.01f) { printf("FAIL [%d]=%f exp=%f\n",i,h_s[i],expected); pass=0; }
    }
    for (int i = 1; i < n; i++) {
        if (h_s[i] >= h_s[i-1]) { printf("FAIL monotonic\n"); pass=0; }
    }

    ocl_free(d_s); ocl_free(d_a);
    printf("Test Decay: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}
