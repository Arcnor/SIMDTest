#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <immintrin.h>

#define SIMD 1

struct argb_pixel {
	uint8_t r,g,b,a;
};

static inline void _darken(const uint32_t *src, uint32_t *dst, uint32_t lightness) {
	argb_pixel *srcPixel = (argb_pixel*)src;
	argb_pixel *dstPixel = (argb_pixel*)dst;

	dstPixel->r = (srcPixel->r * lightness) >> 8;
	dstPixel->g = (srcPixel->g * lightness) >> 8;
	dstPixel->b = (srcPixel->b * lightness) >> 8;
}

static inline void _darkenSIMD(const uint32_t *src, uint32_t *dst, __m128i lightness) {
	argb_pixel *srcPixel = (argb_pixel*)src;
	argb_pixel *dstPixel = (argb_pixel*)dst;

	auto processed = _mm_setr_epi16(
			srcPixel->r, srcPixel->g, srcPixel->b, 0,
			0, 0, 0, 0
	);

	processed = _mm_mullo_epi16(processed, lightness);
	processed = _mm_srli_epi16(processed, 8);

	// TODO: a faster way might be just packing the 16 bit values into 8 bits, then extracting a single 32 bit into dstPixel...
	dstPixel->r = _mm_extract_epi16(processed, 0);
	dstPixel->g = _mm_extract_epi16(processed, 1);
	dstPixel->b = _mm_extract_epi16(processed, 2);
}

int main(int argc, char *argv[]) {
	if (argc < 4) {
		fprintf(stderr, "Usage: %s <image.jpg/png> <output.png> <lightness level [0-255]>", argv[0]);
		return -1;
	}

	auto lightness = std::atoi(argv[3]);

	int width, height, channels;
	uint8_t *img = stbi_load(argv[1], &width, &height, &channels, 4);
	if (img == nullptr) {
		fprintf(stderr, "Cannot load the image\n");
		return -2;
	}

#if SIMD
	auto multiplier = _mm_setr_epi16(
			lightness, lightness, lightness, 0,
			0, 0, 0, 0
	);
#endif

	auto *img32 = reinterpret_cast<uint32_t*>(img);

	// TODO: For the SIMD version, going pixel by pixel is obviously less performant, specially if you can take advantage of AVX or a similar 512 bit architecture
	for (int j = 0; j < width * height; j++) {
#if SIMD
		_darkenSIMD(img32 + j, img32 + j, multiplier);
#else
		_darken(img32 + j, img32 + j, lightness);
#endif
	}

	stbi_write_png(argv[2], width, height, 4, img, width * 4);

	stbi_image_free(img);

	return 0;
}
