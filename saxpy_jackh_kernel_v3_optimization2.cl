#define BLOCK_SIZE 16
#define SIMD_WORK_ITEMS 64
#define n 1000000

__kernel 
__attribute((reqd_work_group_size(BLOCK_SIZE,1,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
__attribute((num_compute_units(1)))
void saxpy(__global const float *x,
           __global const float *y,
           __global float *z,
           const float alpha)
{
    __local float __attribute__((numbanks(1),bankwidth(8))) x_local[BLOCK_SIZE];
    __local float __attribute__((numbanks(1),bankwidth(8))) y_local[BLOCK_SIZE];
    __local float __attribute__((numbanks(1),bankwidth(8))) z_local[BLOCK_SIZE];

    int local_id = get_local_id(0);
    
    #pragma unroll 4
    for (int a = 0; a < n; a += BLOCK_SIZE)
    {
        x_local[local_id] = x[a + local_id];
        y_local[local_id] = y[a + local_id];

        barrier(CLK_LOCAL_MEM_FENCE);

        z_local[local_id] = alpha * x_local[local_id] + y_local[local_id];

        barrier(CLK_LOCAL_MEM_FENCE);

        z[a + local_id] = z_local[local_id];
    }
}
