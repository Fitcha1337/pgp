#ifndef DATASET_H
#define DATASET_H

#include <string>

// Структура для представления датасета
typedef struct Dataset {
    float* images;  // Изображения в формате float32
    int* labels;    // Метки классов
    int num_samples;
    int num_channels;
    int height;
    int width;
} Dataset;

// Функции для работы с датасетом
Dataset load_dataset(const std::string& images_path, const std::string& labels_path);
void free_dataset(Dataset& dataset);

#endif // DATASET_H