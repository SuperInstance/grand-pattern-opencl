//
// test_balance.c — Test balance check kernel (OpenCL)
//

#include "opencl_context.h"
#include <stdio.h>
#include <string.h>

int test_balance(opencl_ctx_t *ctx) {
    printf("=== Test: Balance Check (OpenCL) ===\n");
    const int n = 6;

    cl_uint h_in[n], h_out[n], h_res[n], h_count = 0;
    h_in[0]=5; h_out[0]=5; h_in[1]=10; h_out[1]=10; h_in[2]=3; h_out[2]=3;
    h_in[3]=5; h_out[3]=3; h_in[4]=0; h_out[4]=1; h_in[5]=7; h_out[5]=7;

    cl_mem d_in = ocl_alloc_copy(ctx, h_in, n*sizeof(cl_uint));
    cl_mem d_out = ocl_alloc_copy(ctx, h_out, n*sizeof(cl_uint));
    cl_mem d_res = ocl_alloc(ctx, n*sizeof(cl_uint));
    cl_mem d_count = ocl_alloc_copy(ctx, &h_count, sizeof(cl_uint));

    cl_uint u_n = n;
    clSetKernelArg(ctx->balance_check, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(ctx->balance_check, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(ctx->balance_check, 2, sizeof(cl_mem), &d_res);
    clSetKernelArg(ctx->balance_check, 3, sizeof(cl_mem), &d_count);
    clSetKernelArg(ctx->balance_check, 4, sizeof(cl_uint), &u_n);

    size_t global = n, local = n;
    clEnqueueNDRangeKernel(ctx->queue, ctx->balance_check, 1, NULL, &global, &local, 0, NULL, NULL);
    ocl_finish(ctx);
    ocl_copy_to_host(ctx, h_res, d_res, n*sizeof(cl_uint));
    ocl_copy_to_host(ctx, &h_count, d_count, sizeof(cl_uint));

    int pass = 1;
    if (h_res[0]||h_res[1]||h_res[2]) { printf("FAIL balanced\n"); pass=0; }
    if (!h_res[3]||!h_res[4]) { printf("FAIL imbalanced\n"); pass=0; }
    if (h_res[5]) { printf("FAIL room 5\n"); pass=0; }
    if (h_count!=2) { printf("FAIL count=%u\n",h_count); pass=0; }

    ocl_free(d_in); ocl_free(d_out); ocl_free(d_res); ocl_free(d_count);
    printf("Test Balance Check: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}
