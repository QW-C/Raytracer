#pragma once

#include "Helper.hpp"

namespace raytracer {

class Buffer {
public:
	Buffer(uint32_t size) : size(size) {
		check_cuda_result(::cudaMalloc(&ptr, size));
		check_cuda_result(::cudaMemset(ptr, 0, size));
	}

	~Buffer() {
		check_cuda_result(::cudaFree(ptr));
	}

	Buffer(const Buffer&) = delete;
	Buffer& operator=(const Buffer&) = delete;

	uint32_t get_size() const {
		return size;
	}

	void* get_address() {
		return ptr;
	}

	void upload(const void* data, size_t num_bytes, size_t dst_offset = 0) {
		check_cuda_result(::cudaMemcpy(static_cast<uint8_t*>(ptr) + dst_offset, data, num_bytes, cudaMemcpyHostToDevice));
	}

	template<typename T> void upload_object(const T& data, size_t dst_offset = 0) {
		check_cuda_result(::cudaMemcpy(static_cast<uint8_t*>(ptr) + dst_offset, &data, sizeof(data), cudaMemcpyHostToDevice));
	}

	std::unique_ptr<uint8_t[]> download() {
		auto ret = std::make_unique<uint8_t[]>(size);
		check_cuda_result(::cudaMemcpy(ret.get(), ptr, size, cudaMemcpyDeviceToHost));
		return ret;
	}
private:
	void* ptr = nullptr;
	uint32_t size = 0;
};

class FrameBuffer {
public:
	FrameBuffer(uint32_t width, uint32_t height) :
		width(width),
		height(height),
		data(width * height * sizeof(glm::vec4)) {
	}

	uint32_t get_width() const {
		return width;
	}

	uint32_t get_height() const {
		return height;
	}

	void* get_address() {
		return data.get_address();
	}

	Buffer& get_buffer() {
		return data;
	}
private:
	uint32_t width;
	uint32_t height;
	Buffer data;
};

}