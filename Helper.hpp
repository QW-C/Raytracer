#pragma once

#include "cuda.h"
#include "cuda_runtime_api.h"

#include <chrono>
#include <iostream>
#include <source_location>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/intersect.hpp>

namespace raytracer {

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

inline void check_cuda_result(cudaError_t op, std::source_location source = std::source_location::current()) {
	if(::cudaSuccess != op) {
		std::cerr << "CUDA Error: " << ::cudaGetErrorName(op) << " at: " << source.function_name() << " line: " << source.line() << std::endl;
		__debugbreak();
	}
}

inline void check_cuda_last_error(std::source_location source = std::source_location::current()) {
	if(auto op = ::cudaGetLastError(); ::cudaSuccess != op) {
		std::cerr << "CUDA Last Error: " << ::cudaGetErrorName(op) << " at: " << source.function_name() << " line: " << source.line() << std::endl;
		__debugbreak();
	}
}

class HostTimer {
public:
	HostTimer() {
		reset();
	}

	auto elapsed() const {
		return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - begin);
	}

	void reset() {
		begin = std::chrono::high_resolution_clock::now();
	}
private:
	std::chrono::steady_clock::time_point begin {};
};

}