#include "dataset.h"
#include <iostream>
#include <fstream>
#include <vector>

Dataset load_dataset(const std::string& images_path, const std::string& labels_path) {
    Dataset dataset = {};

    // Открытие файлов
    std::ifstream images_file(images_path, std::ios::binary);
    std::ifstream labels_file(labels_path, std::ios::binary);

    if (!images_file.is_open() || !labels_file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файлы датасета." << std::endl;
        return dataset;
    }

    // Чтение заголовков IDX формата (MNIST files)
    int magic_number = 0, num_images = 0, num_labels = 0, rows = 0, cols = 0;
    images_file.read(reinterpret_cast<char*>(&magic_number), 4);
    images_file.read(reinterpret_cast<char*>(&num_images), 4);
    images_file.read(reinterpret_cast<char*>(&rows), 4);
    images_file.read(reinterpret_cast<char*>(&cols), 4);

    labels_file.read(reinterpret_cast<char*>(&magic_number), 4);
    labels_file.read(reinterpret_cast<char*>(&num_labels), 4);

    // Перевод из Big-Endian в Little-Endian
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);
    num_labels = __builtin_bswap32(num_labels);

    dataset.num_samples = num_images;
    dataset.num_channels = 1;
    dataset.height = rows;
    dataset.width = cols;

    // Выделение памяти
    dataset.images = new float[num_images * rows * cols];
    dataset.labels = new int[num_labels];

    // Чтение данных изображений
    for (int i = 0; i < num_images * rows * cols; ++i) {
        unsigned char pixel = 0;
        images_file.read(reinterpret_cast<char*>(&pixel), 1);
        dataset.images[i] = pixel / 255.0f;  // Нормализация в диапазон [0, 1]
    }

    // Чтение меток
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        labels_file.read(reinterpret_cast<char*>(&label), 1);
        dataset.labels[i] = label;
    }

    images_file.close();
    labels_file.close();

    return dataset;
}

void free_dataset(Dataset& dataset) {
    delete[] dataset.images;
    delete[] dataset.labels;
    dataset.images = nullptr;
    dataset.labels = nullptr;
}