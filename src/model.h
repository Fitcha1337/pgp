#ifndef MODEL_H
#define MODEL_H

#include "dataset.h"

// Структура для представления модели
typedef struct Model {
    // Параметры модели (например, веса) будут добавлены позже
} Model;

// Функции для работы с моделью
Model create_cnn_model();
void train_model(Model* model, const Dataset& train, const Dataset& test, int epochs, int batch_size, float learning_rate);
void save_model(const Model* model, const std::string& output_path);
void free_model(Model* model);

#endif // MODEL_H