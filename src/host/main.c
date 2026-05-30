//
// main.c — OpenCL kernel test runner
// Grand Pattern Fibonacci Dual-Direction Architecture
//

#include "opencl_context.h"
#include <stdio.h>
#include <stdlib.h>

extern int test_cosine_similarity(opencl_ctx_t *ctx);
extern int test_batch_predict(opencl_ctx_t *ctx);
extern int test_balance(opencl_ctx_t *ctx);
extern int test_decay(opencl_ctx_t *ctx);
extern int test_vibe(opencl_ctx_t *ctx);
extern int test_correlation(opencl_ctx_t *ctx);
extern int test_merge(opencl_ctx_t *ctx);
extern int test_e2e(opencl_ctx_t *ctx);

int main(int argc, char **argv) {
    const char *kernel_dir = CL_KERNEL_DIR;
    if (argc > 1) kernel_dir = argv[1];

    printf("Grand Pattern — OpenCL Kernel Tests\n");
    printf("====================================\n");
    printf("Loading kernels from: %s\n\n", kernel_dir);

    opencl_ctx_t ctx;
    if (ocl_init(kernel_dir, &ctx) != 0) {
        fprintf(stderr, "Failed to initialize OpenCL\n");
        fprintf(stderr, "Ensure you have an OpenCL platform and ICD installed.\n");
        return 1;
    }

    int passed = 0, failed = 0;

    if (test_cosine_similarity(&ctx) == 0) passed++; else failed++;
    if (test_batch_predict(&ctx) == 0) passed++; else failed++;
    if (test_balance(&ctx) == 0) passed++; else failed++;
    if (test_decay(&ctx) == 0) passed++; else failed++;
    if (test_vibe(&ctx) == 0) passed++; else failed++;
    if (test_correlation(&ctx) == 0) passed++; else failed++;
    if (test_merge(&ctx) == 0) passed++; else failed++;
    if (test_e2e(&ctx) == 0) passed++; else failed++;

    printf("\n====================================\n");
    printf("Results: %d/8 passed, %d failed\n", passed, failed);

    ocl_cleanup(&ctx);
    return failed > 0 ? 1 : 0;
}
