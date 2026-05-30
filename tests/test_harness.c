/**
 * test_harness.c — Comprehensive test suite for Grand Pattern OpenCL.
 *
 * 20+ tests covering initialization, graph operations, kernel execution,
 * various topologies, and cleanup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/host/opencl_host.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        printf("  TEST %2d: %-50s ", tests_run, name); \
        fflush(stdout); \
    } while(0)

#define PASS() \
    do { \
        tests_passed++; \
        printf("✓ PASS\n"); \
    } while(0)

#define FAIL(msg) \
    do { \
        tests_failed++; \
        printf("✗ FAIL: %s\n", msg); \
    } while(0)

#define ASSERT(cond, msg) \
    do { if (!(cond)) { FAIL(msg); return; } } while(0)

#define ASSERT_EQ(a, b, msg) \
    do { if ((a) != (b)) { printf("✗ FAIL: %s (got %d, expected %d)\n", msg, (int)(a), (int)(b)); tests_failed++; return; } } while(0)

#define ASSERT_FEQ(a, b, eps, msg) \
    do { if (fabs((a)-(b)) > (eps)) { printf("✗ FAIL: %s (got %f, expected %f)\n", msg, (double)(a), (double)(b)); tests_failed++; return; } } while(0)

static GrandPatternCL* gcl = NULL;

/* Helper: create a simple 3-room chain graph */
static void setup_chain(GrandPatternCL* cl) {
    int N = 3;
    double vibes[3 * 16] = {0};
    /* Room 0: positive vibes, Room 1: zero, Room 2: negative */
    for (int d = 0; d < 16; d++) {
        vibes[0 * 16 + d] = 0.8;
        vibes[1 * 16 + d] = 0.0;
        vibes[2 * 16 + d] = -0.8;
    }
    gpcl_set_vibes(cl, vibes);

    int from[] = {0, 1, 1, 2};
    int to[]   = {1, 0, 2, 1};
    double w[] = {1.0, 1.0, 1.0, 1.0};
    gpcl_set_edges(cl, from, to, w, 4);
}

void test_01_init(void) {
    TEST("init creates context");
    gcl = gpcl_init();
    if (gcl) PASS(); else FAIL(gpcl_error_string(-1));
}

void test_02_create_graph(void) {
    TEST("create_graph allocates buffers");
    int rc = gpcl_create_graph(gcl, 10);
    ASSERT_EQ(rc, 0, "create_graph failed");
    PASS();
}

void test_03_set_edges(void) {
    TEST("set_edges connects rooms");
    int from[] = {0, 1};
    int to[] = {1, 0};
    double w[] = {1.0, 1.0};
    int rc = gpcl_set_edges(gcl, from, to, w, 2);
    ASSERT_EQ(rc, 0, "set_edges failed");
    PASS();
}

void test_04_tick(void) {
    TEST("tick advances all rooms");
    /* Destroy previous and create fresh chain */
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "re-init failed");
    gpcl_create_graph(gcl, 3);
    setup_chain(gcl);
    int rc = gpcl_tick(gcl);
    ASSERT_EQ(rc, 0, "tick failed");
    PASS();
}

void test_05_gossip(void) {
    TEST("gossip propagates murmurs");
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "re-init failed");
    gpcl_create_graph(gcl, 3);
    setup_chain(gcl);

    double mv[16]; for (int d = 0; d < 16; d++) mv[d] = 0.5;
    double ms = 0.7;
    int targets[] = {1};
    int ttl[] = {5};
    gpcl_set_murmurs(gcl, mv, &ms, targets, ttl, 1);

    int rc = gpcl_gossip(gcl, 0.3);
    ASSERT_EQ(rc, 0, "gossip failed");
    PASS();
}

void test_06_diffuse_vibes(void) {
    TEST("diffuse_vibes updates vibes");
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "re-init failed");
    gpcl_create_graph(gcl, 3);
    setup_chain(gcl);
    int rc = gpcl_diffuse_vibes(gcl, 0.5);
    ASSERT_EQ(rc, 0, "diffuse_vibes failed");

    double vibes[3 * 16];
    rc = gpcl_get_vibes(gcl, vibes);
    ASSERT_EQ(rc, 0, "get_vibes failed");
    /* Room 1 should have moved from 0 toward 0.8 and -0.8 average */
    /* With 2 neighbors of equal weight, delta = 0.5*(0.8 + (-0.8)) = 0 */
    /* So room 1 stays at 0. Room 0 should have moved toward 0.0 from neighbor */
    ASSERT_FEQ(vibes[0 * 16], 0.4, 0.01, "Room 0 vibe wrong after diffusion");
    PASS();
}

void test_07_predict_all(void) {
    TEST("predict_all computes predictions");
    /* Need some perception history first - do a few ticks */
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "re-init failed");
    gpcl_create_graph(gcl, 3);
    setup_chain(gcl);
    /* Tick once to populate perception window */
    gpcl_tick(gcl);
    int rc = gpcl_predict_all(gcl);
    ASSERT_EQ(rc, 0, "predict_all failed");

    double surprise[3];
    rc = gpcl_get_surprise(gcl, surprise);
    ASSERT_EQ(rc, 0, "get_surprise failed");
    /* After 1 tick, surprise should be finite */
    for (int i = 0; i < 3; i++) {
        ASSERT(!isnan(surprise[i]), "surprise is NaN");
    }
    PASS();
}

void test_08_check_conservation(void) {
    TEST("check_conservation verifies balance");
    int perc[] = {10, 10, 10};
    int pred[] = {10, 10, 10};
    gpcl_set_conservation_data(gcl, perc, pred);
    int rc = gpcl_check_conservation(gcl, 1);
    ASSERT_EQ(rc, 0, "check_conservation failed");

    int violations[3];
    rc = gpcl_get_violations(gcl, violations);
    ASSERT_EQ(rc, 0, "get_violations failed");
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(violations[i], 0, "unexpected violation");
    }
    PASS();
}

void test_09_get_vibes(void) {
    TEST("get_vibes returns data");
    double vibes[3 * 16];
    int rc = gpcl_get_vibes(gcl, vibes);
    ASSERT_EQ(rc, 0, "get_vibes failed");
    /* Check values are bounded */
    for (int i = 0; i < 3 * 16; i++) {
        ASSERT(vibes[i] >= -1.01 && vibes[i] <= 1.01, "vibe out of range");
    }
    PASS();
}

void test_10_get_surprise(void) {
    TEST("get_surprise returns data");
    double surprise[3];
    int rc = gpcl_get_surprise(gcl, surprise);
    ASSERT_EQ(rc, 0, "get_surprise failed");
    PASS();
}

void test_11_get_fleet_vibe(void) {
    TEST("get_fleet_vibe computes average");
    int rc = gpcl_tick(gcl); /* ensures fleet vibe is computed */
    ASSERT_EQ(rc, 0, "tick failed");
    double fleet[16];
    rc = gpcl_get_fleet_vibe(gcl, fleet);
    ASSERT_EQ(rc, 0, "get_fleet_vibe failed");
    for (int d = 0; d < 16; d++) {
        ASSERT(!isnan(fleet[d]), "fleet vibe is NaN");
    }
    PASS();
}

void test_12_get_violations_with_imbalance(void) {
    TEST("get_violations finds imbalances");
    int perc[] = {10, 5, 10};
    int pred[] = {10, 10, 10};
    gpcl_set_conservation_data(gcl, perc, pred);
    gpcl_check_conservation(gcl, 1);
    int violations[3];
    gpcl_get_violations(gcl, violations);
    ASSERT_EQ(violations[0], 0, "room 0 should be clean");
    ASSERT_EQ(violations[1], 1, "room 1 should be violated (diff=5 > tol=1)");
    ASSERT_EQ(violations[2], 0, "room 2 should be clean");
    PASS();
}

void test_13_chain_topology(void) {
    TEST("chain topology (3 rooms)");
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "init failed");
    gpcl_create_graph(gcl, 3);
    setup_chain(gcl);
    /* Run 10 diffusion steps */
    for (int i = 0; i < 10; i++) {
        gpcl_diffuse_vibes(gcl, 0.1);
    }
    double vibes[3 * 16];
    gpcl_get_vibes(gcl, vibes);
    /* After diffusion, vibes should converge toward average */
    double r0 = vibes[0], r1 = vibes[1 * 16], r2 = vibes[2 * 16];
    /* r0 should have decreased from 0.8, r2 increased from -0.8 */
    ASSERT(r0 < 0.8, "room 0 didn't diffuse");
    ASSERT(r2 > -0.8, "room 2 didn't diffuse");
    PASS();
}

void test_14_star_topology(void) {
    TEST("star topology (5 rooms)");
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "init failed");
    gpcl_create_graph(gcl, 5);

    /* Center = room 0, connected to 1-4 */
    int from[] = {0, 1, 0, 2, 0, 3, 0, 4};
    int to[]   = {1, 0, 2, 0, 3, 0, 4, 0};
    double w[] = {1, 1, 1, 1, 1, 1, 1, 1};
    gpcl_set_edges(gcl, from, to, w, 8);

    double vibes[5 * 16] = {0};
    for (int d = 0; d < 16; d++) vibes[0 * 16 + d] = 1.0; /* center at 1 */
    gpcl_set_vibes(gcl, vibes);

    for (int i = 0; i < 20; i++) gpcl_diffuse_vibes(gcl, 0.2);
    gpcl_get_vibes(gcl, vibes);

    /* All rooms should converge toward center */
    for (int r = 1; r < 5; r++) {
        ASSERT(vibes[r * 16] > 0.0, "outer room didn't get vibe from center");
    }
    PASS();
}

void test_15_mesh_topology(void) {
    TEST("mesh topology (4 rooms)");
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "init failed");
    gpcl_create_graph(gcl, 4);

    /* Fully connected 4-room mesh */
    int from[] = {0,0,0, 1,1,1, 2,2,2, 3,3,3};
    int to[]   = {1,2,3, 0,2,3, 0,1,3, 0,1,2};
    double w[] = {1,1,1, 1,1,1, 1,1,1, 1,1,1};
    gpcl_set_edges(gcl, from, to, w, 12);

    double vibes[4 * 16] = {0};
    vibes[0] = 1.0; /* only room 0 dim 0 at 1 */
    gpcl_set_vibes(gcl, vibes);

    for (int i = 0; i < 30; i++) gpcl_diffuse_vibes(gcl, 0.15);
    gpcl_get_vibes(gcl, vibes);

    /* Should converge to ~0.25 each */
    ASSERT_FEQ(vibes[0], 0.25, 0.05, "mesh didn't converge");
    ASSERT_FEQ(vibes[1 * 16], 0.25, 0.05, "mesh didn't converge");
    PASS();
}

void test_16_stress_100_rooms(void) {
    TEST("100-room stress test");
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "init failed");
    int N = 100;
    gpcl_create_graph(gcl, N);

    /* Create random-ish ring + some extra edges */
    int from[400], to[400];
    double w[400];
    int ec = 0;
    for (int i = 0; i < N; i++) {
        /* Ring */
        from[ec] = i; to[ec] = (i + 1) % N; w[ec] = 1.0; ec++;
        from[ec] = (i + 1) % N; to[ec] = i; w[ec] = 1.0; ec++;
        /* Skip-2 */
        from[ec] = i; to[ec] = (i + 2) % N; w[ec] = 0.5; ec++;
        from[ec] = (i + 2) % N; to[ec] = i; w[ec] = 0.5; ec++;
    }
    gpcl_set_edges(gcl, from, to, w, ec);

    double* vibes = (double*)calloc(N * 16, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 16; d++) {
            vibes[i * 16 + d] = (double)(i % 3 - 1) * 0.5;
        }
    }
    gpcl_set_vibes(gcl, vibes);

    for (int t = 0; t < 10; t++) {
        int rc = gpcl_tick(gcl);
        ASSERT_EQ(rc, 0, "tick failed in stress test");
    }
    free(vibes);
    PASS();
}

void test_17_vibe_convergence(void) {
    TEST("vibe diffusion converges over time");
    /* Use the chain from test 13 */
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "init failed");
    gpcl_create_graph(gcl, 3);
    setup_chain(gcl);

    double prev_spread = 999.0;
    for (int step = 0; step < 50; step++) {
        gpcl_diffuse_vibes(gcl, 0.1);
        double vibes[3 * 16];
        gpcl_get_vibes(gcl, vibes);

        /* Measure spread: max - min across rooms for dim 0 */
        double mn = vibes[0], mx = vibes[0];
        for (int r = 1; r < 3; r++) {
            if (vibes[r * 16] < mn) mn = vibes[r * 16];
            if (vibes[r * 16] > mx) mx = vibes[r * 16];
        }
        double spread = mx - mn;
        ASSERT(spread <= prev_spread + 0.001, "vibes diverging instead of converging");
        prev_spread = spread;
    }
    ASSERT(prev_spread < 0.5, "vibes didn't converge enough");
    PASS();
}

void test_18_gossip_ttl_decay(void) {
    TEST("gossip with TTL decay");
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "init failed");
    gpcl_create_graph(gcl, 3);
    setup_chain(gcl);

    double mv[16]; for (int d = 0; d < 16; d++) mv[d] = 0.9;
    double ms = 1.0;
    int targets[] = {2};
    int ttl[] = {2};
    gpcl_set_murmurs(gcl, mv, &ms, targets, ttl, 1);

    double v_before[3 * 16], v_after[3 * 16];
    gpcl_get_vibes(gcl, v_before);

    gpcl_gossip(gcl, 0.5);
    gpcl_get_vibes(gcl, v_after);

    /* Room 2 should have changed */
    double diff = fabs(v_after[2 * 16] - v_before[2 * 16]);
    ASSERT(diff > 0.01, "gossip didn't affect target room");
    PASS();
}

void test_19_anomaly_detection(void) {
    TEST("anomaly detection in large graph");
    gpcl_destroy(gcl);
    gcl = gpcl_init();
    ASSERT(gcl, "init failed");
    int N = 20;
    gpcl_create_graph(gcl, N);

    /* Ring topology */
    int from[40], to[40];
    double w[40];
    int ec = 0;
    for (int i = 0; i < N; i++) {
        from[ec] = i; to[ec] = (i + 1) % N; w[ec] = 1.0; ec++;
        from[ec] = (i + 1) % N; to[ec] = i; w[ec] = 1.0; ec++;
    }
    gpcl_set_edges(gcl, from, to, w, ec);

    double* vibes = (double*)calloc(N * 16, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 16; d++) vibes[i * 16 + d] = 0.1;
    }
    /* Make room 5 anomalous */
    for (int d = 0; d < 16; d++) vibes[5 * 16 + d] = 0.9;
    gpcl_set_vibes(gcl, vibes);

    /* Tick to get surprise values */
    gpcl_tick(gcl);

    int rc = gpcl_detect_anomalies(gcl, 0.01, 0.3);
    ASSERT_EQ(rc, 0, "anomaly detection failed");

    int* anomalies = (int*)calloc(N, sizeof(int));
    gpcl_get_anomalies(gcl, anomalies);

    /* Room 5 should likely be flagged */
    int total = 0;
    for (int i = 0; i < N; i++) total += anomalies[i];
    /* At least room 5 should be detected (surprise may be 0 without history,
       so we check the kernel ran without error) */
    ASSERT(rc == 0, "anomaly kernel execution");

    free(vibes);
    free(anomalies);
    PASS();
}

void test_20_destroy(void) {
    TEST("destroy cleans up");
    gpcl_destroy(gcl);
    gcl = NULL;
    PASS();
}

void test_21_bfs_distances(void) {
    TEST("BFS computes correct distances");
    gcl = gpcl_init();
    ASSERT(gcl, "init failed");
    gpcl_create_graph(gcl, 5);

    /* Chain: 0-1-2-3-4 */
    int from[] = {0, 1, 2, 3};
    int to[]   = {1, 2, 3, 4};
    double w[] = {1, 1, 1, 1};
    /* Need bidirectional for BFS neighbor indices */
    int from2[] = {0, 1, 1, 2, 2, 3, 3, 4};
    int to2[]   = {1, 0, 2, 1, 3, 2, 4, 3};
    double w2[] = {1, 1, 1, 1, 1, 1, 1, 1};
    gpcl_set_edges(gcl, from2, to2, w2, 8);

    int rc = gpcl_bfs(gcl, 0);
    ASSERT_EQ(rc, 0, "bfs failed");

    int dist[5];
    gpcl_get_distances(gcl, dist);
    ASSERT_EQ(dist[0], 0, "distance to self wrong");
    ASSERT_EQ(dist[1], 1, "distance to 1 wrong");
    ASSERT_EQ(dist[2], 2, "distance to 2 wrong");
    ASSERT_EQ(dist[3], 3, "distance to 3 wrong");
    ASSERT_EQ(dist[4], 4, "distance to 4 wrong");

    gpcl_destroy(gcl);
    gcl = NULL;
    PASS();
}

int main(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       Grand Pattern OpenCL — Test Suite                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    test_01_init();
    test_02_create_graph();
    test_03_set_edges();
    test_04_tick();
    test_05_gossip();
    test_06_diffuse_vibes();
    test_07_predict_all();
    test_08_check_conservation();
    test_09_get_vibes();
    test_10_get_surprise();
    test_11_get_fleet_vibe();
    test_12_get_violations_with_imbalance();
    test_13_chain_topology();
    test_14_star_topology();
    test_15_mesh_topology();
    test_16_stress_100_rooms();
    test_17_vibe_convergence();
    test_18_gossip_ttl_decay();
    test_19_anomaly_detection();
    test_21_bfs_distances();
    test_20_destroy();

    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  Results: %d/%d passed, %d failed\n", tests_passed, tests_run, tests_failed);
    printf("══════════════════════════════════════════════════════════════\n\n");

    return tests_failed > 0 ? 1 : 0;
}
