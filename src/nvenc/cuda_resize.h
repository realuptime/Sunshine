/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __CUDA_RESIZE_H__
#define __CUDA_RESIZE_H__

#include <cstdint>

/**
 * The imageFormat enum is used to identify the pixel format and colorspace
 * of an image.  Supported data types are based on `uint8` and `float`, with
 * colorspaces including RGB/RGBA, BGR/BGRA, grayscale, YUV, and Bayer.
 *
 * There are also a variety of helper functions available that provide info about
 * each format at runtime - for example, the pixel bit depth (imageFormatDepth())
 * the number of image channels (imageFormatChannels()), and computing the size of
 * an image from it's dimensions (@see imageFormatSize()).  To convert between
 * image formats using the GPU, there is also the cudaConvertColor() function.
 *
 * In addition to the enums below, each format can also be identified by a string.
 * The string corresponding to each format is included in the documentation below.
 * These strings are more commonly used from Python, but can also be used from C++
 * with the imageFormatFromStr() and imageFormatToStr() functions.
 *
 * @ingroup imageFormat
 */
enum imageFormat
{
	// RGB
	IMAGE_RGB8=0,					/**< uchar3 RGB8    (`'rgb8'`) */
	IMAGE_RGBA8,					/**< uchar4 RGBA8   (`'rgba8'`) */
	IMAGE_RGB32F,					/**< float3 RGB32F  (`'rgb32f'`) */
	IMAGE_RGBA32F,					/**< float4 RGBA32F (`'rgba32f'`) */

	// BGR
	IMAGE_BGR8,					/**< uchar3 BGR8    (`'bgr8'`) */
	IMAGE_BGRA8,					/**< uchar4 BGRA8   (`'bgra8'`) */
	IMAGE_BGR32F,					/**< float3 BGR32F  (`'bgr32f'`) */
	IMAGE_BGRA32F,					/**< float4 BGRA32F (`'bgra32f'`) */
	
	// YUV
	IMAGE_YUYV,					/**< YUV YUYV 4:2:2 packed (`'yuyv'`) */
	IMAGE_YUY2=IMAGE_YUYV,			/**< Duplicate of YUYV     (`'yuy2'`) */
	IMAGE_YVYU,					/**< YUV YVYU 4:2:2 packed (`'yvyu'`) */
	IMAGE_UYVY,					/**< YUV UYVY 4:2:2 packed (`'uyvy'`) */
	IMAGE_I420,					/**< YUV I420 4:2:0 planar (`'i420'`) */
	IMAGE_YV12,					/**< YUV YV12 4:2:0 planar (`'yv12'`) */
	IMAGE_NV12,					/**< YUV NV12 4:2:0 planar (`'nv12'`) */
	
	// Bayer
	IMAGE_BAYER_BGGR,				/**< 8-bit Bayer BGGR (`'bayer-bggr'`) */
	IMAGE_BAYER_GBRG,				/**< 8-bit Bayer GBRG (`'bayer-gbrg'`) */
	IMAGE_BAYER_GRBG,				/**< 8-bit Bayer GRBG (`'bayer-grbg'`) */
	IMAGE_BAYER_RGGB,				/**< 8-bit Bayer RGGB (`'bayer-rggb'`) */
	
	// grayscale
	IMAGE_GRAY8,					/**< uint8 grayscale  (`'gray8'`)   */
	IMAGE_GRAY32F,					/**< float grayscale  (`'gray32f'`) */

	// extras
	IMAGE_COUNT,					/**< The number of image formats */
	IMAGE_UNKNOWN=999,				/**< Unknown/undefined format */
	IMAGE_DEFAULT=IMAGE_RGBA32F		/**< Default format (IMAGE_RGBA32F) */
};

/**
 * Enumeration of interpolation filtering modes.
 * @see cudaFilterModeFromStr() and cudaFilterModeToStr()
 * @ingroup cudaFilter
 */
enum cudaFilterMode
{
	FILTER_POINT,	 /**< Nearest-neighbor sampling */
	FILTER_LINEAR	 /**< Bilinear filtering */
};


/**
 * Enumeration of image layout formats.
 * @ingroup cudaFilter
 */
enum cudaDataFormat
{
	FORMAT_HWC,	/**< Height * Width * Channels (packed format) */
	FORMAT_CHW,	/**< Channels * Width * Height (DNN format) */
	
	/**< Default format (HWC) */
	FORMAT_DEFAULT = FORMAT_HWC
};

#ifdef __CUDACC__

/**
 * Rescale a uint8 grayscale image on the GPU.
 * To use bilinear filtering for upscaling, set filter to FILTER_LINEAR.
 * If the image is being downscaled, or if FILTER_POINT is set (default),
 * then nearest-neighbor sampling will be used instead.
 * @ingroup resize
 */
cudaError_t cudaResize( uint8_t* input,  size_t inputWidth,  size_t inputHeight,
				    uint8_t* output, size_t outputWidth, size_t outputHeight,
				    cudaFilterMode filter=FILTER_POINT );

/**
 * Rescale a floating-point grayscale image on the GPU.
 * To use bilinear filtering for upscaling, set filter to FILTER_LINEAR.
 * If the image is being downscaled, or if FILTER_POINT is set (default),
 * then nearest-neighbor sampling will be used instead.
 * @ingroup resize
 */
cudaError_t cudaResize( float* input,  size_t inputWidth,  size_t inputHeight,
				    float* output, size_t outputWidth, size_t outputHeight,
				    cudaFilterMode filter=FILTER_POINT );

/**
 * Rescale a uchar3 RGB/BGR image on the GPU.
 * To use bilinear filtering for upscaling, set filter to FILTER_LINEAR.
 * If the image is being downscaled, or if FILTER_POINT is set (default),
 * then nearest-neighbor sampling will be used instead.
 * @ingroup resize
 */
cudaError_t cudaResize( uchar3* input,  size_t inputWidth,  size_t inputHeight,
				    uchar3* output, size_t outputWidth, size_t outputHeight,
				    cudaFilterMode filter=FILTER_POINT );

/**
 * Rescale a float3 RGB/BGR image on the GPU.
 * To use bilinear filtering for upscaling, set filter to FILTER_LINEAR.
 * If the image is being downscaled, or if FILTER_POINT is set (default),
 * then nearest-neighbor sampling will be used instead.
 * @ingroup resize
 */
cudaError_t cudaResize( float3* input,  size_t inputWidth,  size_t inputHeight,
				    float3* output, size_t outputWidth, size_t outputHeight,
				    cudaFilterMode filter=FILTER_POINT );

/**
 * Rescale a uchar4 RGBA/BGRA image on the GPU.
 * To use bilinear filtering for upscaling, set filter to FILTER_LINEAR.
 * If the image is being downscaled, or if FILTER_POINT is set (default),
 * then nearest-neighbor sampling will be used instead.
 * @ingroup resize
 */
cudaError_t cudaResize( uchar4* input,  size_t inputWidth,  size_t inputHeight,
				    uchar4* output, size_t outputWidth, size_t outputHeight,
				    cudaFilterMode filter=FILTER_POINT );

/**
 * Rescale a float4 RGBA/BGRA image on the GPU.
 * To use bilinear filtering for upscaling, set filter to FILTER_LINEAR.
 * If the image is being downscaled, or if FILTER_POINT is set (default),
 * then nearest-neighbor sampling will be used instead.
 * @ingroup resize
 */
cudaError_t cudaResize( float4* input,  size_t inputWidth,  size_t inputHeight,
				    float4* output, size_t outputWidth, size_t outputHeight,
				    cudaFilterMode filter=FILTER_POINT );
#else
typedef CUresult cudaError_t;
#endif // __CUDACC__

/**
 * Rescale an image on the GPU (supports grayscale, RGB/BGR, RGBA/BGRA)
 * To use bilinear filtering for upscaling, set filter to FILTER_LINEAR.
 * If the image is being downscaled, or if FILTER_POINT is set (default),
 * then nearest-neighbor sampling will be used instead.
 * @ingroup resize
 */
cudaError_t cudaResize( void* input,  size_t inputWidth,  size_t inputHeight,
				    void* output, size_t outputWidth, size_t outputHeight, 
				    imageFormat format, cudaFilterMode filter=FILTER_POINT );

#endif
