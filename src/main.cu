#include <iostream>
#include "dataset.h"
#include "model.h"

int main() {
    // Загрузка данных
    Dataset train = load_dataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    Dataset test = load_dataset("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    // Создание модели CNN
    Model model = create_cnn_model();

    // Параметры обучения
    int epochs = 5;
    int batch_size = 64;
    float learning_rate = 0.001;

    // Обучение модели
    train_model(&model, train, test, epochs, batch_size, learning_rate);

    // Сохранение модели
    save_model(&model, "build/model_weights.bin");

    // Освобождение памяти
    free_dataset(train);
    free_dataset(test);
    free_model(&model);

    return 0;
}