//
// test_cosine.c — Test cosine similarity kernel (OpenCL)
//

#include "opencl_context.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

int test_cosine_similarity(opencl_ctx_t *ctx) {
    printf("=== Test: Cosine Similarity (OpenCL) ===\n");
    const int n = 4, dim = 8;
    cl_int err;

    float h_a[n*dim], h_b[n*dim], h_r[n];
    for (int j = 0; j < dim; j++) { h_a[j]=1; h_b[j]=1; }
    for (int j = 0; j < dim; j++) { h_a[dim+j]=(j<4)?1:0; h_b[dim+j]=(j<4)?0:1; }
    for (int j = 0; j < dim; j++) { h_a[2*dim+j]=1; h_b[2*dim+j]=-1; }
    for (int j = 0; j < dim; j++) { h_a[3*dim+j]=(float)(j+1); h_b[3*dim+j]=(float)(j+2); }

    cl_mem d_a = ocl_alloc_copy(ctx, h_a, n*dim*sizeof(float));
    cl_mem d_b = ocl_alloc_copy(ctx, h_b, n*dim*sizeof(float));
    cl_mem d_r = ocl_alloc(ctx, n*sizeof(float));

    cl_uint u_n = n, u_dim = dim;
    clSetKernelArg(ctx->cosine_similarity, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(ctx->cosine_similarity, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(ctx->cosine_similarity, 2, sizeof(cl_mem), &d_r);
    clSetKernelArg(ctx->cosine_similarity, 3, sizeof(cl_uint), &u_n);
    clSetKernelArg(ctx->cosine_similarity, 4, sizeof(cl_uint), &u_dim);

    size_t global = n * 256, local = 256;
    clEnqueueNDRangeKernel(ctx->queue, ctx->cosine_similarity, 1, NULL, &global, &local, 0, NULL, NULL);
    ocl_finish(ctx);
    ocl_copy_to_host(ctx, h_r, d_r, n*sizeof(float));

    int pass = 1;
    if (fabsf(h_r[0]-1.0f)>0.01f) { printf("FAIL pair 0: %f\n",h_r[0]); pass=0; }
    if (fabsf(h_r[1])>0.01f) { printf("FAIL pair 1: %f\n",h_r[1]); pass=0; }
    if (fabsf(h_r[2]+1.0f)>0.01f) { printf("FAIL pair 2: %f\n",h_r[2]); pass=0; }
    printf("Pair 3: %f\n", h_r[3]);

    ocl_free(d_a); ocl_free(d_b); ocl_free(d_r);
    printf("Test Cosine Similarity: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}
