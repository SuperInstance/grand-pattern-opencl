//
// kernels.cl — OpenCL kernel implementations
// Grand Pattern Fibonacci Dual-Direction Architecture
//

// Cosine similarity between pairs of embeddings
// Each work-group handles one pair
__kernel void cosine_similarity(
    __global const float* a,
    __global const float* b,
    __global float* result,
    const uint n,
    const uint dim)
{
    uint pair_id = get_group_id(0);
    if (pair_id >= n) return;

    uint tid = get_local_id(0);
    uint block_size = get_local_size(0);

    __global const float* a_ptr = a + pair_id * dim;
    __global const float* b_ptr = b + pair_id * dim;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (uint i = tid; i < dim; i += block_size) {
        float va = a_ptr[i];
        float vb = b_ptr[i];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    // Local memory reduction
    __local float s_dot[256], s_na[256], s_nb[256];
    s_dot[tid] = dot;
    s_na[tid] = norm_a;
    s_nb[tid] = norm_b;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dot[tid] += s_dot[tid + stride];
            s_na[tid] += s_na[tid + stride];
            s_nb[tid] += s_nb[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        float denom = sqrt(s_na[0]) * sqrt(s_nb[0]);
        result[pair_id] = (denom > 1e-8f) ? s_dot[0] / denom : 0.0f;
    }
}

// Batch predict: prediction = perception + delta * (target - perception)
__kernel void batch_predict(
    __global const float* perceptions,
    __global const float* targets,
    __global float* result,
    const uint n,
    const uint dim,
    const float delta)
{
    uint room = get_group_id(0);
    if (room >= n) return;

    uint tid = get_local_id(0);
    uint block_size = get_local_size(0);

    __global const float* p = perceptions + room * dim;
    __global const float* t = targets + room * dim;
    __global float* r = result + room * dim;

    for (uint i = tid; i < dim; i += block_size) {
        r[i] = p[i] + delta * (t[i] - p[i]);
    }
}

// Balance check: each thread checks one room
__kernel void balance_check(
    __global const uint* z_in,
    __global const uint* z_out,
    __global uint* result,
    __global uint* imbalance_count,
    const uint n)
{
    uint i = get_global_id(0);
    if (i >= n) return;

    if (z_in[i] != z_out[i]) {
        result[i] = 1;
        atomic_inc(imbalance_count);
    } else {
        result[i] = 0;
    }
}

// Decay: strengths[i] *= exp(-rate * ages[i])
__kernel void decay(
    __global float* strengths,
    __global const float* ages,
    const uint n,
    const float rate)
{
    uint i = get_global_id(0);
    if (i >= n) return;

    strengths[i] *= exp(-rate * ages[i]);
}

// Vibe compute: vibe = normalize(embedding + velocity * dt)
__kernel void vibe_compute(
    __global const float* embeddings,
    __global const float* velocities,
    __global float* vibes,
    const uint n,
    const uint dim,
    const float dt)
{
    uint room = get_group_id(0);
    if (room >= n) return;

    uint tid = get_local_id(0);
    uint block_size = get_local_size(0);

    __global const float* emb = embeddings + room * dim;
    __global const float* vel = velocities + room * dim;
    __global float* vib = vibes + room * dim;

    float norm_sq = 0.0f;
    for (uint i = tid; i < dim; i += block_size) {
        float v = emb[i] + vel[i] * dt;
        vib[i] = v;
        norm_sq += v * v;
    }

    // Local memory reduction
    __local float s_norm[256];
    s_norm[tid] = norm_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_norm[tid] += s_norm[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float norm = sqrt(s_norm[0]);
    if (norm > 1e-8f) {
        for (uint i = tid; i < dim; i += block_size) {
            vib[i] /= norm;
        }
    }
}

// Correlation matrix: all-pairs cosine similarity
__kernel void correlation_matrix(
    __global const float* vibes,
    __global float* matrix,
    const uint n_rooms,
    const uint dim)
{
    uint row = get_global_id(0);
    uint col = get_global_id(1);
    if (row >= n_rooms || col >= n_rooms) return;
    if (row > col) return;

    __global const float* a = vibes + row * dim;
    __global const float* b = vibes + col * dim;

    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float va = a[i], vb = b[i];
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }

    float denom = sqrt(na) * sqrt(nb);
    float sim = (denom > 1e-8f) ? dot / denom : 0.0f;

    matrix[row * n_rooms + col] = sim;
    matrix[col * n_rooms + row] = sim;
}

// Merge candidates: identify consecutive pairs above threshold
__kernel void merge_candidates(
    __global const float* embeddings,
    __global const float* strengths,
    __global uint* candidates,
    __global uint* candidate_count,
    const uint n,
    const uint dim,
    const float threshold)
{
    uint i = get_global_id(0);
    if (i >= n - 1) return;

    __global const float* a = embeddings + i * dim;
    __global const float* b = embeddings + (i + 1) * dim;

    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (uint j = 0; j < dim; j++) {
        float va = a[j], vb = b[j];
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }

    float denom = sqrt(na) * sqrt(nb);
    float sim = (denom > 1e-8f) ? dot / denom : 0.0f;

    if (sim >= threshold) {
        candidates[i] = 1;
        atomic_inc(candidate_count);
    } else {
        candidates[i] = 0;
    }
}
