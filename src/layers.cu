#include "layers.h"
#include <iostream>
#include <cuda_runtime.h>

// CUDA-ядро для обратного распространения в Conv2D
__global__ void conv2d_backward_kernel(const float* input, const float* grad_output, const float* weights,
                                       float* grad_input, float* grad_weights, float* grad_biases,
                                       int input_height, int input_width, int input_channels,
                                       int kernel_size, int stride, int padding,
                                       int output_height, int output_width, int num_filters) {
    int filter_idx = blockIdx.z;                       // Индекс фильтра
    int out_x = blockIdx.x * blockDim.x + threadIdx.x; // X-координата выходного пикселя
    int out_y = blockIdx.y * blockDim.y + threadIdx.y; // Y-координата выходного пикселя

    if (out_x < output_width && out_y < output_height && filter_idx < num_filters) {
        int output_idx = ((filter_idx * output_height + out_y) * output_width + out_x);
        float grad_out_val = grad_output[output_idx];

        // Обновляем градиенты смещений
        atomicAdd(&grad_biases[filter_idx], grad_out_val);

        for (int c = 0; c < input_channels; ++c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_x = out_x * stride + kx - padding;
                    int in_y = out_y * stride + ky - padding;

                    if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                        int input_idx = ((c * input_height + in_y) * input_width + in_x);
                        int weight_idx = (((filter_idx * input_channels + c) * kernel_size + ky) * kernel_size + kx);

                        // Обновляем градиенты весов
                        atomicAdd(&grad_weights[weight_idx], grad_out_val * input[input_idx]);

                        // Распространяем градиенты на вход
                        atomicAdd(&grad_input[input_idx], grad_out_val * weights[weight_idx]);
                    }
                }
            }
        }
    }
}

// Обратное распространение для Conv2D
void backward_conv2d(const Conv2DLayer& layer, const float* input, const float* grad_output, float* grad_input, float* grad_weights, float* grad_biases) {
    int output_height = (input_height - layer.kernel_size + 2 * layer.padding) / layer.stride + 1;
    int output_width = (input_width - layer.kernel_size + 2 * layer.padding) / layer.stride + 1;

    dim3 block_size(16, 16, 1);
    dim3 grid_size((output_width + block_size.x - 1) / block_size.x,
                   (output_height + block_size.y - 1) / block_size.y,
                   layer.num_filters);

    // Инициализация градиентов нулями
    cudaMemset(grad_weights, 0, sizeof(float) * layer.num_filters * input_channels * layer.kernel_size * layer.kernel_size);
    cudaMemset(grad_biases, 0, sizeof(float) * layer.num_filters);
    cudaMemset(grad_input, 0, sizeof(float) * input_height * input_width * input_channels);

    // Вызов CUDA-ядра
    conv2d_backward_kernel<<<grid_size, block_size>>>(
        input, grad_output, layer.weights, grad_input, grad_weights, grad_biases,
        input_height, input_width, input_channels, layer.kernel_size, layer.stride,
        layer.padding, output_height, output_width, layer.num_filters);

    cudaDeviceSynchronize();
}

// CUDA-ядро для обратного распространения в Dense
__global__ void dense_backward_kernel(const float* input, const float* grad_output, const float* weights,
                                      float* grad_input, float* grad_weights, float* grad_biases,
                                      int input_size, int output_size) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_idx < output_size) {
        float grad_out_val = grad_output[neuron_idx];

        // Обновляем градиенты смещений
        atomicAdd(&grad_biases[neuron_idx], grad_out_val);

        for (int i = 0; i < input_size; ++i) {
            // Обновляем градиенты весов
            atomicAdd(&grad_weights[neuron_idx * input_size + i], grad_out_val * input[i]);

            // Распространяем градиенты на вход
            atomicAdd(&grad_input[i], grad_out_val * weights[neuron_idx * input_size + i]);
        }
    }
}

// Обратное распространение для Dense слоя
void backward_dense(const DenseLayer& layer, const float* input, const float* grad_output, float* grad_input, float* grad_weights, float* grad_biases) {
    dim3 block_size(256);
    dim3 grid_size((layer.output_size + block_size.x - 1) / block_size.x);

    // Инициализация градиентов нулями
    cudaMemset(grad_weights, 0, sizeof(float) * layer.input_size * layer.output_size);
    cudaMemset(grad_biases, 0, sizeof(float) * layer.output_size);
    cudaMemset(grad_input, 0, sizeof(float) * layer.input_size);

    // Вызов CUDA-ядра
    dense_backward_kernel<<<grid_size, block_size>>>(
        input, grad_output, layer.weights, grad_input, grad_weights, grad_biases,
        layer.input_size, layer.output_size);

    cudaDeviceSynchronize();
}