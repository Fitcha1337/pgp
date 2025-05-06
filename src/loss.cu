#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

// CUDA-ядро для вычисления градиента Cross-Entropy Loss
__global__ void compute_loss_gradient_kernel(const float* predictions, const int* labels, float* grad_output, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        int label = labels[idx];
        for (int c = 0; c < num_classes; ++c) {
            int output_idx = idx * num_classes + c;
            grad_output[output_idx] = (c == label ? predictions[output_idx] - 1.0f : predictions[output_idx]);
        }
    }
}

// Функция для вызова CUDA-ядра
void compute_loss_gradient(const float* predictions, const int* labels, float* grad_output, int batch_size, int num_classes) {
    dim3 block_size(256);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x);

    // Вызов CUDA-ядра
    compute_loss_gradient_kernel<<<grid_size, block_size>>>(predictions, labels, grad_output, batch_size, num_classes);

    // Синхронизация устройства
    cudaDeviceSynchronize();
}