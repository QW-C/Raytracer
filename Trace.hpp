#include "Buffer.hpp"
#include "Scene.hpp"

namespace raytracer {

struct RaytracingParameters {
	FrameBuffer& output;
	uint32_t w;
	uint32_t h;
	uint32_t num_samples;

	Scene& scene;
	Buffer& gpu_scene;
};

void trace_rays(RaytracingParameters& parameters, Buffer& out);

}