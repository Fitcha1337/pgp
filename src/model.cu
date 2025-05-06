#include "loss.cu" // Подключение файла с реализацией функции compute_loss_gradient

void train_model(Model* model, const Dataset& train, const Dataset& test, int epochs, int batch_size, float learning_rate) {
    std::cout << "Начало обучения модели." << std::endl;

    int num_batches = train.num_samples / batch_size;

    // Выделение памяти для хранения выходов, градиентов и промежуточных данных
    float* conv1_output = allocate_memory(batch_size * model->conv1.num_filters * model->conv1.kernel_size * model->conv1.kernel_size);
    float* dense1_output = allocate_memory(batch_size * model->dense1.output_size);
    float* dense2_output = allocate_memory(batch_size * model->dense2.output_size);

    float* grad_dense2_output = allocate_memory(batch_size * model->dense2.output_size);
    float* grad_dense1_output = allocate_memory(batch_size * model->dense1.output_size);
    float* grad_conv1_output = allocate_memory(batch_size * model->conv1.num_filters * model->conv1.kernel_size * model->conv1.kernel_size);

    // Цикл по эпохам
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Эпоха " << epoch + 1 << " из " << epochs << "..." << std::endl;

        // Цикл по батчам
        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // Получение текущего батча данных
            float* batch_images = train.images + batch_idx * batch_size * train.width * train.height * train.num_channels;
            int* batch_labels = train.labels + batch_idx * batch_size;

            // Прямое распространение
            forward_conv2d(model->conv1, batch_images, conv1_output, train.height, train.width);
            forward_dense(model->dense1, conv1_output, dense1_output);
            forward_dense(model->dense2, dense1_output, dense2_output);

            // Вычисление градиентов на выходе (Cross-Entropy Loss)
            compute_loss_gradient(dense2_output, batch_labels, grad_dense2_output, batch_size, model->dense2.output_size);

            // Обратное распространение
            backward_dense(model->dense2, dense1_output, grad_dense2_output, grad_dense1_output, model->dense2.weights, model->dense2.biases);
            backward_dense(model->dense1, conv1_output, grad_dense1_output, grad_conv1_output, model->dense1.weights, model->dense1.biases);
            backward_conv2d(model->conv1, batch_images, grad_conv1_output, train.height, train.width, model->conv1.weights, model->conv1.biases);

            // Обновление параметров
            update_parameters(model->dense2.weights, model->dense2.biases, model->dense2.grad_weights, model->dense2.grad_biases,
                              model->dense2.input_size * model->dense2.output_size, model->dense2.output_size, learning_rate);
            update_parameters(model->dense1.weights, model->dense1.biases, model->dense1.grad_weights, model->dense1.grad_biases,
                              model->dense1.input_size * model->dense1.output_size, model->dense1.output_size, learning_rate);
            update_parameters(model->conv1.weights, model->conv1.biases, model->conv1.grad_weights, model->conv1.grad_biases,
                              model->conv1.num_filters * model->conv1.kernel_size * model->conv1.kernel_size * train.num_channels, model->conv1.num_filters, learning_rate);
        }

        std::cout << "Эпоха " << epoch + 1 << " завершена." << std::endl;
    }

    // Освобождение памяти
    cudaFree(conv1_output);
    cudaFree(dense1_output);
    cudaFree(dense2_output);
    cudaFree(grad_dense2_output);
    cudaFree(grad_dense1_output);
    cudaFree(grad_conv1_output);

    std::cout << "Обучение завершено." << std::endl;
}