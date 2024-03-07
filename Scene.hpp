#pragma once

#include "Buffer.hpp"
#include "Helper.hpp"
#include <memory>
#include <vector>

namespace raytracer {

struct Sphere {
	glm::vec3 center;
	float radius;
	glm::vec4 colour;
};

struct Camera {
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 up;
	glm::vec3 right;
	float aspect_ratio;
};

class Scene {
public:
	Scene() = default;
	~Scene() = default;

	void set_camera(const Camera& camera) {
		scene_camera = camera;
	}

	Camera& get_camera() {
		return scene_camera;
	}

	void add_sphere(const Sphere& sphere) {
		spheres.push_back(sphere);
	}

	Buffer& upload_scene() {
		unsigned size = static_cast<unsigned>(sizeof(Sphere) * spheres.size());
		gpu_scene = std::make_unique<Buffer>(size);
		gpu_scene->upload(&spheres[0], size);
		return *gpu_scene;
	}
private:
	Camera scene_camera {};
	std::vector<Sphere> spheres;

	std::unique_ptr<Buffer> gpu_scene;
};

}