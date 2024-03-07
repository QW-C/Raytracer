#include "Buffer.hpp"
#include "Helper.hpp"
#include "Image.hpp"
#include "Scene.hpp"
#include "Trace.hpp"

namespace {

void populate_scene(const glm::uvec3& render_dimensions, raytracer::Scene& scene) {
	{
		glm::vec3 position {0.f, 0.f, 0.f};
		glm::vec3 direction = {0.f, 0.f, 1.f};
		glm::vec3 right = glm::normalize(glm::cross({0.f, 1.f, 0.f}, direction));

		raytracer::Camera camera {
			.position = position,
			.direction = direction,
			.up = glm::normalize(glm::cross(direction, right)),
			.right = right,
			.aspect_ratio = static_cast<float>(render_dimensions.x) / render_dimensions.y
		};
		scene.set_camera(camera);
	}

	{
		scene.add_sphere({
			.center = {0.f, 0.f, 4.f},
			.radius = 1.f,
			.colour = {1.f, 0.f, 0.f, 1.f}
		});
		scene.add_sphere({
			.center = {2.f, 0.f, 4.f},
			.radius = 1.f,
			.colour = {1.f, 0.f, 1.f, 1.f}
		});
		scene.add_sphere({
			.center = {-3.f, -.5f, 4.f},
			.radius = .5f,
			.colour = {1.f, 1.f, 0.f, 1.f}
		});
		scene.add_sphere({
			.center = {0.f, -101.f, 5.f},
			.radius = 100.f,
			.colour = {.5f, .5f, .5f, 1.f}
		});
		scene.add_sphere({
			.center = {0.f, 5.f, 8.f},
			.radius = 4.f,
			.colour = {.1f, .1f, .1f, 1.f}
		});
	}
}

}

int main() {
	raytracer::check_cuda_result(cudaSetDevice(0));

	{
		glm::uvec3 render_dimensions {800, 600, 1};
		unsigned num_samples = 100;

		raytracer::FrameBuffer render_target(render_dimensions.x, render_dimensions.y);
		raytracer::Buffer out(render_dimensions.x * render_dimensions.y * sizeof(unsigned));

		raytracer::Scene scene;
		populate_scene(render_dimensions, scene);
		auto& gpu_scene = scene.upload_scene();

		raytracer::RaytracingParameters parameters {
			.output = render_target,
			.w = render_dimensions.x,
			.h = render_dimensions.y,
			.num_samples = num_samples,
			.scene = scene,
			.gpu_scene = gpu_scene
		};

		raytracer::trace_rays(parameters, out);

		auto pixel_buffer = out.download();
		raytracer::write_png_u8("output.png", render_dimensions.x, render_dimensions.y, pixel_buffer.get());
	}

	raytracer::check_cuda_result(::cudaDeviceReset());

	return 0;
}