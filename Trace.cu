#include "Trace.cuh"
#include "Trace.hpp"

#include "device_launch_parameters.h"

namespace raytracer {

__device__ float make_random(curandState& state) {
	return curand_uniform(&state) * 2.f - 1.f;
}

__device__ bool first_hit(const Ray& ray, const Sphere* spheres, unsigned num_spheres, HitInfo& hit_info) {
	constexpr float offset = 1e-4f;

	bool found_hit = false;
	float closest = FLT_MAX;
	for(unsigned i = 0; i < num_spheres; ++i) {

		auto& sphere = spheres[i];

		if(float distance = 0.f; glm::intersectRaySphere(ray.origin, ray.direction, sphere.center, sphere.radius * sphere.radius, distance)) {
			if(distance < closest && distance > offset) {

				glm::vec3 position = ray.origin + distance * ray.direction;
				glm::vec3 normal = (position - sphere.center) / sphere.radius;

				if(glm::dot(normal, ray.direction) < 0.f) {
					found_hit = true;
					closest = distance;

					hit_info.position = position;
					hit_info.distance = distance;
					hit_info.normal = normal;
					hit_info.scene_index = i;
				}
				else {
					return false;
				}

			}
		}
	}

	return found_hit;
}

__device__ glm::vec4 traverse_scene(Ray ray, const Sphere* spheres, unsigned num_spheres, curandState& rand_state) {
	glm::vec4 colour {0.f, 0.f, 0.f, 1.f};

	const unsigned num_bounces = 100;
	float occlusion_factor = 1.f;
	for(unsigned i = 0; i < num_bounces; ++i) {
		if(HitInfo hit {}; first_hit(ray, spheres, num_spheres, hit)) {

			ray.origin = hit.position;
			
			ray.direction = glm::normalize(glm::vec3(make_random(rand_state), make_random(rand_state), make_random(rand_state)));
			float orientation = glm::dot(ray.direction, hit.normal) >= 0.f ? 1.f : -1.f;
			ray.direction = glm::normalize(orientation * ray.direction + hit.normal);

			const auto& sphere_colour = spheres[hit.scene_index].colour;
			colour = glm::mix(colour, sphere_colour, occlusion_factor);
			occlusion_factor *= .5f;
		}

		else {
			break;
		}
	}

	colour = glm::clamp(occlusion_factor * colour, 0.f, 1.f);
	colour.a = 1.f;
	return colour;
}

__global__ void launch_rays(LaunchRaysParameters args) {
	unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= args.w || y >= args.h) {
		return;
	}
	unsigned index = x + y * args.w;
	auto& output = args.output[index];
	curandState rand_state;
	curand_init(index, 0, 0, &rand_state);

	glm::vec4 colour {};
	for(int i = 0; i < args.num_samples; ++i) {

		glm::vec2 uv {
			(float(x) + make_random(rand_state)) / args.w,
			(float(y) + make_random(rand_state)) / args.h
		};

		glm::vec3 screen {
			(2.f * uv.x - 1.f) * args.camera.aspect_ratio,
			-(2.f * uv.y - 1.f),
			1.f
		};

		Ray camera_ray {
			args.camera.position,
			glm::normalize(screen.x * args.camera.right + screen.y * args.camera.up + screen.z * args.camera.direction)
		};

		colour += traverse_scene(camera_ray, args.spheres, args.num_spheres, rand_state);
	}
	output = colour / float(args.num_samples);
}

__global__ void f32_to_uint(LaunchToUINTParameters args) {
	unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index >= args.size) {
		return;
	}

	unsigned colour = 0;
	#pragma unroll
	for(int i = 0; i < 4; ++i) {
		unsigned as_uint = __float2uint_rd(pow(args.src[index][i], .5f) * 255.f);
		colour |=  as_uint << (i * 8);
	}

	args.dst[index] = colour;
}

void trace_rays(RaytracingParameters& parameters, Buffer& out) {
	{
		constexpr unsigned num_threads_fullscreen = 8;
		unsigned aligned_x = (parameters.w + num_threads_fullscreen - 1) / num_threads_fullscreen;
		unsigned aligned_y = (parameters.h + num_threads_fullscreen - 1) / num_threads_fullscreen;
		dim3 launch_dimensions_2d {aligned_x, aligned_y, 1};
		dim3 block_2d = {num_threads_fullscreen, num_threads_fullscreen, 1};

		LaunchRaysParameters dispatch_args {
			.w = parameters.w,
			.h = parameters.h,
			.num_samples = parameters.num_samples,
			.output = static_cast<glm::vec4*>(parameters.output.get_address()),
			.camera = parameters.scene.get_camera(),

			.spheres = static_cast<Sphere*>(parameters.gpu_scene.get_address()),
			.num_spheres = parameters.gpu_scene.get_size() / (uint32_t)sizeof(Sphere)
		};

		HostTimer timer;
		launch_rays <<<launch_dimensions_2d, block_2d>>> (dispatch_args);
		raytracer::check_cuda_last_error();
		raytracer::check_cuda_result(cudaDeviceSynchronize());
		std::cout << "Returned in " << timer.elapsed() << std::endl;
	}

	{
		constexpr unsigned num_threads_1d = 32;
		dim3 num_blocks {(out.get_size() + num_threads_1d - 1) / num_threads_1d, 1, 1};
		dim3 block_1d = {num_threads_1d, 1, 1};

		LaunchToUINTParameters to_uint_args {
			.size = out.get_size() / sizeof(unsigned),
			.src = static_cast<glm::vec4*>(parameters.output.get_address()),
			.dst = static_cast<unsigned*>(out.get_address())
		};
		f32_to_uint <<<num_blocks, block_1d>>> (to_uint_args);
		raytracer::check_cuda_last_error();
		raytracer::check_cuda_result(cudaDeviceSynchronize());
	}
}

}
