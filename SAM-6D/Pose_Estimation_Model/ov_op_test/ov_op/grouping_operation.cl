//gather operation kernel



__kernel void grouping_kernel(
    __global const float* features,  // (B, C, N)
    __global const int* idx,         // (B, NPOINT, NSAMPLE)
    __global float* output           // (B, C, NPOINT, NSAMPLE)
) {
    int global_id = get_global_id(0);
    int total = B * C * NPOINT * NSAMPLE;

    if (global_id >= total) return;

    int sample = global_id % NSAMPLE;
    int point = (global_id / NSAMPLE) % NPOINT;
    int channel = (global_id / (NSAMPLE * NPOINT)) % C;
    int batch = global_id / (C * NPOINT * NSAMPLE);

    int idx_offset = batch * (NPOINT * NSAMPLE) + point * NSAMPLE + sample;
    int a = idx[idx_offset];

    float val = 0.0f;

    if (a >= 0 && a < N) {
        int input_offset = batch * (C * N) + channel * N + a;
        val = features[input_offset];

        if (isnan(val) || isinf(val)) {
            val = 0.0f;
        }
    }

    int out_offset = batch * (C * NPOINT * NSAMPLE) + channel * (NPOINT * NSAMPLE) + point * NSAMPLE + sample;
    output[out_offset] = val;


    // Debug print for first few threads
    //if (global_id < 5) {
        //printf("b=%d, c=%d, point=%d, sample=%d, idx=%d, val=%f\n", batch, channel, point, sample, a, val);
    //}
}

