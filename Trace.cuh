#include "Helper.hpp"
#include "Scene.hpp"

#include "curand.h"
#include "curand_kernel.h"

namespace raytracer {

struct HitInfo {
	glm::vec3 position;
	float distance;
	glm::vec3 normal;
	unsigned scene_index;
};

struct LaunchRaysParameters {
	unsigned w;
	unsigned h;
	unsigned num_samples;
	glm::vec4* output;

	Camera camera;

	Sphere* spheres;
	unsigned num_spheres;
};

struct LaunchToUINTParameters {
	size_t size;
	glm::vec4* src;
	unsigned* dst;
};

__device__ float make_random(curandState& state);
__device__ bool first_hit(const Ray& ray, const Sphere* spheres, unsigned num_spheres, HitInfo& hit_info);
__device__ glm::vec4 traverse_scene(Ray, const Sphere* spheres, unsigned num_spheres, curandState& rand_state);

__global__ void launch_rays(LaunchRaysParameters args);
__global__ void f32_to_uint(LaunchToUINTParameters args);

}