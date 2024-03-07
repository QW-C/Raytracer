#include "Image.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace raytracer {

void write_png_u8(const std::string& path, unsigned w, unsigned h, void* pixels) {
	::stbi_write_png(path.c_str(), w, h, 4, pixels, sizeof(unsigned) * w);
}

}