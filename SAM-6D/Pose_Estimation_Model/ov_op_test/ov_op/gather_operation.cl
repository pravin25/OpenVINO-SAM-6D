// gather_opration kernel 

__kernel void gather_kernel(
    __global const INPUT0_TYPE* features,  // (B, C, N)
    __global const INPUT1_TYPE* idx,       // (B, NPOINT)
    __global OUTPUT0_TYPE* output          // (B, C, NPOINT)
) 
{
    int B = INPUT0_DIMS[0];
    int C = INPUT0_DIMS[1];
    int N = INPUT0_DIMS[2];
    int NPOINT = INPUT1_DIMS[1];

    int global_id = get_global_id(0);
    int total = B * C * NPOINT;
    if (global_id >= total)
        return;

    int point = global_id % NPOINT;
    int channel = (global_id / NPOINT) % C;
    int batch = global_id / (C * NPOINT);

    // index find from idx[b, point]
    int idx_offset = batch * NPOINT + point;
    int gather_index = idx[idx_offset];

    OUTPUT0_TYPE val = (OUTPUT0_TYPE)0;
    if (gather_index >= 0 && gather_index < N) {
        int feature_offset = batch * (C * N) + channel * N + gather_index;
        val = features[feature_offset];

        if (isnan(val) || isinf(val)) {
            val = (OUTPUT0_TYPE)0;
        }
    }

    // write output[b, c, point]
    int output_offset = batch * (C * NPOINT) + channel * NPOINT + point;
    output[output_offset] = val;
    
    // debug 
    //if (global_id == 0) {
    //    printf("Gather kernel info: B=%d, C=%d, N=%d, NPOINT=%d\n", B, C, N, NPOINT);
    //    printf("First output value: %f\n", (float)val);
    //}
}

