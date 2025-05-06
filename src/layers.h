#ifndef LAYERS_H
#define LAYERS_H

#include "dataset.h"

// Остальные структуры и функции...

// Функция обновления параметров
void update_parameters(float* weights, float* biases, float* grad_weights, float* grad_biases, 
                       int weight_size, int bias_size, float learning_rate);

#endif // LAYERS_H