//grouping operation kernel

__kernel void grouping_kernel(
    __global const INPUT0_TYPE* features,  // (B, C, N)
    __global const INPUT1_TYPE* idx,       // (B, NPOINT, NSAMPLE)
    __global OUTPUT0_TYPE* output          // (B, C, NPOINT, NSAMPLE)
) {
    // input shapes from OpenVINO runtime metadata
    int B = INPUT0_DIMS[0];
    int C = INPUT0_DIMS[1];
    int N = INPUT0_DIMS[2];

    int NPOINT = INPUT1_DIMS[1];
    int NSAMPLE = INPUT1_DIMS[2];

    int global_id = get_global_id(0);
    int total = B * C * NPOINT * NSAMPLE;
    if (global_id >= total) return;

    // global_id to multidimensional indices
    int sample = global_id % NSAMPLE;
    int point = (global_id / NSAMPLE) % NPOINT;
    int channel = (global_id / (NSAMPLE * NPOINT)) % C;
    int batch = global_id / (C * NPOINT * NSAMPLE);

    // index find
    int idx_offset = batch * (NPOINT * NSAMPLE) + point * NSAMPLE + sample;
    int group_index = idx[idx_offset];

    float val = 0.0f;
    if (group_index >= 0 && group_index < N) {
        int input_offset = batch * (C * N) + channel * N + group_index;
        val = features[input_offset];

        // for safer sanitization
        if (isnan(val) || isinf(val)) {
            val = 0.0f;
            printf("\n catched!..");
        }
    }

    // write output
    int out_offset = batch * (C * NPOINT * NSAMPLE) + channel * (NPOINT * NSAMPLE) + point * NSAMPLE + sample;
    output[out_offset] = val;

    // debug
    // if (global_id == 0) {
    //     printf("Dynamic Grouping kernel: B=%d, C=%d, N=%d, NPOINT=%d, NSAMPLE=%d\n", B, C, N, NPOINT, NSAMPLE);
    // }
}
