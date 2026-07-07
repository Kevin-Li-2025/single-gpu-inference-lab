NVCC ?= nvcc
CUDA_ARCH ?= 89

BUILD_DIR := build
TARGET := $(BUILD_DIR)/sparse_penalty_bench
SRC := sparse_repetition_penalty.cu

NVCCFLAGS ?= -O3 -std=c++17 --use_fast_math -lineinfo
GENCODE := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)

.PHONY: all clean quick

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $< -o $@

quick: $(TARGET)
	./$(TARGET) --quick

clean:
	rm -rf $(BUILD_DIR)
