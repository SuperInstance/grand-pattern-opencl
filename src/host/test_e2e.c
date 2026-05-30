//
// test_e2e.c — End-to-end test: tick → predict → balance → GC (OpenCL)
//

#include "opencl_context.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

int test_e2e(opencl_ctx_t *ctx) {
    printf("=== Test: End-to-End OpenCL (Tick → Predict → Balance → GC) ===\n");
    const int nr = 4, dim = 8;

    float h_emb[nr*dim], h_vel[nr*dim], h_pred[nr*dim], h_vibes[nr*dim];
    float h_str[nr], h_ages[nr];
    cl_uint h_zin[nr], h_zout[nr], h_bal[nr], h_imb = 0;
    cl_uint h_mc[nr], h_mcount = 0;

    for (int i = 0; i < nr; i++) {
        h_str[i] = 1.0f; h_ages[i] = 0.0f;
        h_zin[i] = 5; h_zout[i] = 5;
        for (int j = 0; j < dim; j++) {
            h_emb[i*dim+j] = sinf((float)(i*dim+j));
            h_vel[i*dim+j] = 0.1f * cosf((float)(i*dim+j));
            h_pred[i*dim+j] = h_emb[i*dim+j] + 0.5f;
        }
    }
    h_zout[2] = 3;

    size_t eb = nr*dim*sizeof(float);
    cl_mem d_emb = ocl_alloc_copy(ctx, h_emb, eb);
    cl_mem d_vel = ocl_alloc_copy(ctx, h_vel, eb);
    cl_mem d_pred = ocl_alloc_copy(ctx, h_pred, eb);
    cl_mem d_vibes = ocl_alloc(ctx, eb);
    cl_mem d_str = ocl_alloc_copy(ctx, h_str, nr*sizeof(float));
    cl_mem d_ages = ocl_alloc_copy(ctx, h_ages, nr*sizeof(float));
    cl_mem d_zin = ocl_alloc_copy(ctx, h_zin, nr*sizeof(cl_uint));
    cl_mem d_zout = ocl_alloc_copy(ctx, h_zout, nr*sizeof(cl_uint));
    cl_mem d_bal = ocl_alloc(ctx, nr*sizeof(cl_uint));
    cl_mem d_imb = ocl_alloc_copy(ctx, &h_imb, sizeof(cl_uint));
    cl_mem d_mc = ocl_alloc(ctx, nr*sizeof(cl_uint));
    cl_mem d_mcount = ocl_alloc_copy(ctx, &h_mcount, sizeof(cl_uint));

    // Step 1: Decay
    cl_uint u_nr = nr, u_dim = dim;
    float rate = 0.1f;
    clSetKernelArg(ctx->decay, 0, sizeof(cl_mem), &d_str);
    clSetKernelArg(ctx->decay, 1, sizeof(cl_mem), &d_ages);
    clSetKernelArg(ctx->decay, 2, sizeof(cl_uint), &u_nr);
    clSetKernelArg(ctx->decay, 3, sizeof(float), &rate);
    size_t g1 = nr, l1 = nr;
    clEnqueueNDRangeKernel(ctx->queue, ctx->decay, 1, NULL, &g1, &l1, 0, NULL, NULL);

    // Step 2: Predict
    float delta = 0.5f;
    clSetKernelArg(ctx->batch_predict, 0, sizeof(cl_mem), &d_emb);
    clSetKernelArg(ctx->batch_predict, 1, sizeof(cl_mem), &d_pred);
    clSetKernelArg(ctx->batch_predict, 2, sizeof(cl_mem), &d_pred);
    clSetKernelArg(ctx->batch_predict, 3, sizeof(cl_uint), &u_nr);
    clSetKernelArg(ctx->batch_predict, 4, sizeof(cl_uint), &u_dim);
    clSetKernelArg(ctx->batch_predict, 5, sizeof(float), &delta);
    size_t g2 = nr*256, l2 = 256;
    clEnqueueNDRangeKernel(ctx->queue, ctx->batch_predict, 1, NULL, &g2, &l2, 0, NULL, NULL);

    // Step 3: Vibe
    float dt = 1.0f;
    clSetKernelArg(ctx->vibe_compute, 0, sizeof(cl_mem), &d_emb);
    clSetKernelArg(ctx->vibe_compute, 1, sizeof(cl_mem), &d_vel);
    clSetKernelArg(ctx->vibe_compute, 2, sizeof(cl_mem), &d_vibes);
    clSetKernelArg(ctx->vibe_compute, 3, sizeof(cl_uint), &u_nr);
    clSetKernelArg(ctx->vibe_compute, 4, sizeof(cl_uint), &u_dim);
    clSetKernelArg(ctx->vibe_compute, 5, sizeof(float), &dt);
    clEnqueueNDRangeKernel(ctx->queue, ctx->vibe_compute, 1, NULL, &g2, &l2, 0, NULL, NULL);

    // Step 4: Balance
    clSetKernelArg(ctx->balance_check, 0, sizeof(cl_mem), &d_zin);
    clSetKernelArg(ctx->balance_check, 1, sizeof(cl_mem), &d_zout);
    clSetKernelArg(ctx->balance_check, 2, sizeof(cl_mem), &d_bal);
    clSetKernelArg(ctx->balance_check, 3, sizeof(cl_mem), &d_imb);
    clSetKernelArg(ctx->balance_check, 4, sizeof(cl_uint), &u_nr);
    clEnqueueNDRangeKernel(ctx->queue, ctx->balance_check, 1, NULL, &g1, &l1, 0, NULL, NULL);

    // Step 5: Merge
    float threshold = 0.9f;
    clSetKernelArg(ctx->merge_candidates, 0, sizeof(cl_mem), &d_emb);
    clSetKernelArg(ctx->merge_candidates, 1, sizeof(cl_mem), &d_str);
    clSetKernelArg(ctx->merge_candidates, 2, sizeof(cl_mem), &d_mc);
    clSetKernelArg(ctx->merge_candidates, 3, sizeof(cl_mem), &d_mcount);
    clSetKernelArg(ctx->merge_candidates, 4, sizeof(cl_uint), &u_nr);
    clSetKernelArg(ctx->merge_candidates, 5, sizeof(cl_uint), &u_dim);
    clSetKernelArg(ctx->merge_candidates, 6, sizeof(float), &threshold);
    clEnqueueNDRangeKernel(ctx->queue, ctx->merge_candidates, 1, NULL, &g1, &l1, 0, NULL, NULL);

    ocl_finish(ctx);

    // Read back
    ocl_copy_to_host(ctx, h_str, d_str, nr*sizeof(float));
    ocl_copy_to_host(ctx, h_vibes, d_vibes, eb);
    ocl_copy_to_host(ctx, h_bal, d_bal, nr*sizeof(cl_uint));
    ocl_copy_to_host(ctx, &h_imb, d_imb, sizeof(cl_uint));

    int pass = 1;
    for (int i = 0; i < nr; i++) {
        if (fabsf(h_str[i]-1.0f)>0.01f) { printf("FAIL str[%d]=%f\n",i,h_str[i]); pass=0; }
    }
    if (h_imb!=1) { printf("FAIL imb=%u\n",h_imb); pass=0; }
    if (!h_bal[2]) { printf("FAIL room 2\n"); pass=0; }
    for (int i = 0; i < nr; i++) {
        float norm = 0;
        for (int j = 0; j < dim; j++) norm += h_vibes[i*dim+j]*h_vibes[i*dim+j];
        if (fabsf(norm-1.0f)>0.01f) { printf("FAIL vibe norm[%d]=%f\n",i,norm); pass=0; }
    }

    ocl_free(d_emb); ocl_free(d_vel); ocl_free(d_pred); ocl_free(d_vibes);
    ocl_free(d_str); ocl_free(d_ages); ocl_free(d_zin); ocl_free(d_zout);
    ocl_free(d_bal); ocl_free(d_imb); ocl_free(d_mc); ocl_free(d_mcount);

    printf("Test End-to-End: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}
