#pragma once

#include <string>

namespace raytracer {

void write_png_u8(const std::string& path, unsigned w, unsigned h, void* pixels);

}