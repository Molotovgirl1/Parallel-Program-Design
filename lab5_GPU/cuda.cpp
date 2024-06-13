#include <stdio.h>
void ReSet(float* data_D, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++)
            data_D[i * N + j] = 0;
        data_D[i * N + i] = 1.0;
        for (int j = i + 1; j < N; j++)
            data_D[i * N + j] = rand() % 100;
    }
    for (int i = 0; i < N; i++) {
        int k1 = rand() % N;
        int k2 = rand() % N;
        for (int j = 0; j < N; j++) {
            data_D[i * N + j] += data_D[j];
            data_D[k1 * N + j] += data_D[k2 * N + j];
        }
    }
}
__global__ void division_kernel(float* data, int k, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid + 1; i < N; i += stride) {
        data[k * N + i] = (float)(data[k * N + i] / data[k * N + k]);
    }
    return;
}
__global__ void eliminate_kernel(float* data, int k, int N) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx == 0) data[k * N + k] = 1.0;
    int row = k + 1 + blockIdx.x;
    while (row < N) {
        int tid = threadIdx.x;
        while (k + 1 + tid < N) {
            int col = k + 1 + tid;
            float temp_1 = data[(row * N) + col];
            float temp_2 = data[(row * N) + k];
            float temp_3 = data[k * N + col];
            data[(row * N) + col] = temp_1 - temp_2 * temp_3;
            tid = tid + blockDim.x;
        }
        __syncthreads();
        if (threadIdx.x == 0) data[row * N + k] = 0;
        row += gridDim.x;
    }
    return;
}
int main() {
    int grid = 256;
    int block = 1024;
    int N = 2000;
    float* data_D;
    int size = (N + 1) * (N + 1);
    cudaMallocManaged(&data_D, size);
    ReSet(data_D, N);
    cudaDeviceSynchronize();
    cudaError_t ret;
    for (int k = 0; k < N; k++) {
        division_kernel << <grid, block >> > (data_D, k, N);
        cudaDeviceSynchronize();
        ret = cudaGetLastError();
        if (ret != cudaSuccess) {
            printf("division_kernel failed,%s\n", cudaGetErrorString(ret));
        }
        eliminate_kernel << <grid, block >> > (data_D, k, N);
        cudaDeviceSynchronize();
        ret = cudaGetLastError();
        if (ret != cudaSuccess) {
            printf("eliminate_kernel failed,%s\n", cudaGetErrorString(ret));
        }
    }
    cudaFree(data_D);
    return 0;
}