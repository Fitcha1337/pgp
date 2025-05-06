# Папки
SRC_DIR = src
BUILD_DIR = build
DATA_DIR = data

# Компилятор и флаги
NVCC = nvcc
CXXFLAGS = -std=c++11 -Xcompiler -Wall -O2

# Список исходных файлов
SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/dataset.cu $(SRC_DIR)/model.cu $(SRC_DIR)/layers.cu $(SRC_DIR)/loss.cu
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SOURCES))
EXEC = $(BUILD_DIR)/cnn

# Цель по умолчанию
all: init $(EXEC)

# Правило сборки исполняемого файла
$(EXEC): $(OBJS)
	$(NVCC) $(CXXFLAGS) -o $@ $^

# Правило сборки объектных файлов
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CXXFLAGS) -c -o $@ $<

# Инициализация папки build
init:
	mkdir -p $(BUILD_DIR)

# Очистка
clean:
	rm -rf $(BUILD_DIR)/*.o $(EXEC)

# Запуск
run: all
	./$(EXEC)

# Убедиться, что данные существуют
check-data:
	@if [ ! -d "$(DATA_DIR)" ]; then \
		echo "Отсутствует папка данных $(DATA_DIR). Создайте её и добавьте необходимые файлы."; \
		exit 1; \
	fi;

# Полная очистка
dist-clean: clean
	rm -rf $(BUILD_DIR)