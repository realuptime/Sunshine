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

#include "cuda_resize.h"
#include "cuda_utility.h"
#include "cuda_math.h"

#include "cuda.h"

// imageFormatDepth
size_t imageFormatDepth( imageFormat format )
{
	switch(format)
	{
		case IMAGE_RGB8:		
		case IMAGE_BGR8:		return sizeof(uchar3) * 8;
		case IMAGE_RGBA8:		
		case IMAGE_BGRA8:		return sizeof(uchar4) * 8;
		case IMAGE_RGB32F:		
		case IMAGE_BGR32F:		return sizeof(float3) * 8;
		case IMAGE_RGBA32F: 	
		case IMAGE_BGRA32F:		return sizeof(float4) * 8;
		case IMAGE_GRAY8:		return sizeof(unsigned char) * 8;
		case IMAGE_GRAY32F:		return sizeof(float) * 8;
		case IMAGE_I420:
		case IMAGE_YV12:
		case IMAGE_NV12:		return 12;
		case IMAGE_UYVY:
		case IMAGE_YUYV:		
		case IMAGE_YVYU:		return 16;
		case IMAGE_BAYER_BGGR:
		case IMAGE_BAYER_GBRG:
		case IMAGE_BAYER_GRBG:
		case IMAGE_BAYER_RGGB:	return sizeof(unsigned char) * 8;
	}

	return 0;
}



size_t imageFormatSize( imageFormat format, size_t width, size_t height )
{
    return (width * height * imageFormatDepth(format)) / 8;
}

/**
 * cudaCheckError
 * @ingroup cudaError
 */
cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line)
{
#if !defined(CUDA_TRACE)
        if( retval == cudaSuccess)
                return cudaSuccess;
#endif

        //int activeDevice = -1;
        //cudaGetDevice(&activeDevice);

        //Log("[cuda]   device %i  -  %s\n", activeDevice, txt);

        if( retval == cudaSuccess )
        {
                printf("CUDA %s\n", txt);
        }
        else
        {
                printf("CUDA %s\n", txt);
        }

        if( retval != cudaSuccess )
        {
                printf("CUDA ERR:   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
                printf("CUDA ERR   %s:%i\n", file, line);
        }

        return retval;
}



/**
 * CUDA device function for reading a pixel from an image, either in HWC or CHW layout.
 *
 * @param input pointer to image in CUDA device memory
 * @param x desired x-coordinate to sample
 * @param y desired y-coordinate to sample
 * @param width width of the input image
 * @param height height of the input image
 *
 * @returns the raw pixel data from the input image
 */
template<cudaDataFormat layout, typename T>
__device__ inline T cudaReadPixel( T* input, int x, int y, int width, int height )
{
	return input[y * width + x];
}

template<> __device__ inline 
float2 cudaReadPixel<FORMAT_CHW>( float2* input, int x, int y, int width, int height )
{
	float* ptr = (float*)input;
	const int offset = y * width + x;
	return make_float2(ptr[offset], ptr[width * height + offset]);
}

template<> __device__ inline 
float3 cudaReadPixel<FORMAT_CHW>( float3* input, int x, int y, int width, int height )
{
	float* ptr = (float*)input;
	const int offset = y * width + x;
	const int pixels = width * height;
	return make_float3(ptr[offset], ptr[pixels + offset], ptr[pixels * 2 + offset]);
}

template<> __device__ inline 
float4 cudaReadPixel<FORMAT_CHW>( float4* input, int x, int y, int width, int height )
{
	float* ptr = (float*)input;
	const int offset = y * width + x;
	const int pixels = width * height;
	return make_float4(ptr[offset], ptr[pixels + offset], ptr[pixels * 2 + offset], ptr[pixels * 3 + offset]);
}



/**
 * CUDA device function for sampling a pixel with bilinear or point filtering.
 * cudaFilterPixel() is for use inside of other CUDA kernels, and accepts a
 * cudaFilterMode template parameter which sets the filtering mode, in addition
 * to a cudaDataFormat template parameter which sets the format (HWC or CHW).
 *
 * @param input pointer to image in CUDA device memory
 * @param x desired x-coordinate to sample
 * @param y desired y-coordinate to sample
 * @param width width of the input image
 * @param height height of the input image
 *
 * @returns the filtered pixel from the input image
 * @ingroup cudaFilter
 */ 
template<cudaFilterMode filter, cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel( T* input, float x, float y, int width, int height )
{
	if( filter == FILTER_POINT )
	{
		const int x1 = int(x);
		const int y1 = int(y);

		return cudaReadPixel<format>(input, x1, y1, width, height); //input[y1 * width + x1];
	}
	else // FILTER_LINEAR
	{
		const float bx = x - 0.5f;
		const float by = y - 0.5f;

		const float cx = bx < 0.0f ? 0.0f : bx;
		const float cy = by < 0.0f ? 0.0f : by;

		const int x1 = int(cx);
		const int y1 = int(cy);
			
		const int x2 = x1 >= width - 1 ? x1 : x1 + 1;	// bounds check
		const int y2 = y1 >= height - 1 ? y1 : y1 + 1;
		
		const T samples[4] = {
			cudaReadPixel<format>(input, x1, y1, width, height),   //input[y1 * width + x1],
			cudaReadPixel<format>(input, x2, y1, width, height),   //input[y1 * width + x2],
			cudaReadPixel<format>(input, x1, y2, width, height),   //input[y2 * width + x1],
			cudaReadPixel<format>(input, x2, y2, width, height) }; //input[y2 * width + x2] };

		// compute bilinear weights
		const float x1d = cx - float(x1);
		const float y1d = cy - float(y1);

		const float x1f = 1.0f - x1d;
		const float y1f = 1.0f - y1d;

		const float x2f = 1.0f - x1f;
		const float y2f = 1.0f - y1f;

		const float x1y1f = x1f * y1f;
		const float x1y2f = x1f * y2f;
		const float x2y1f = x2f * y1f;
		const float x2y2f = x2f * y2f;

		return samples[0] * x1y1f + samples[1] * x2y1f + samples[2] * x1y2f + samples[3] * x2y2f;
	}
}

/**
 * CUDA device function for sampling a pixel with bilinear or point filtering.
 * cudaFilterPixel() is for use inside of other CUDA kernels, and samples a
 * pixel from an input image from the scaled coordinates of an output image.
 *
 * @param input pointer to image in CUDA device memory
 * @param x desired x-coordinate to sample (in coordinate space of output image)
 * @param y desired y-coordinate to sample (in coordinate space of output image)
 * @param input_width width of the input image
 * @param input_height height of the input image
 * @param output_width width of the output image
 * @param output_height height of the output image
 *
 * @returns the filtered pixel from the input image
 * @ingroup cudaFilter
 */ 
template<cudaFilterMode filter, cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel( T* input, int x, int y,
						       int input_width, int input_height,
						       int output_width, int output_height )
{
	const float px = float(x) / float(output_width) * float(input_width);
	const float py = float(y) / float(output_height) * float(input_height);

	return cudaFilterPixel<filter, format>(input, px, py, input_width, input_height);
}




// gpuResize
template<typename T, cudaFilterMode filter>
__global__ void gpuResize( T* input, int inputWidth, int inputHeight, T* output, int outputWidth, int outputHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= outputWidth || y >= outputHeight )
		return;

	output[y * outputWidth + x] = cudaFilterPixel<filter>(input, x, y, inputWidth, inputHeight, outputWidth, outputHeight); 
}

// launchResize
template<typename T>
static cudaError_t launchResize( T* input, size_t inputWidth, size_t inputHeight,
				             T* output, size_t outputWidth, size_t outputHeight,
						   cudaFilterMode filter )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	if( outputWidth < inputWidth && outputHeight < inputHeight )
		filter = FILTER_POINT;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	#define launch_resize(filterMode)	\
		gpuResize<T, filterMode><<<gridDim, blockDim>>>(input, inputWidth, inputHeight, output, outputWidth, outputHeight)
	
	if( filter == FILTER_POINT )
		launch_resize(FILTER_POINT);
	else if( filter == FILTER_LINEAR )
		launch_resize(FILTER_LINEAR);

	return CUDA(cudaGetLastError());
}

// cudaResize (uint8 grayscale)
cudaError_t cudaResize( uint8_t* input, size_t inputWidth, size_t inputHeight, uint8_t* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter )
{
	return launchResize<uint8_t>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter);
}

// cudaResize (float grayscale)
cudaError_t cudaResize( float* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter )
{
	return launchResize<float>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter);
}

// cudaResize (uchar3)
cudaError_t cudaResize( uchar3* input, size_t inputWidth, size_t inputHeight, uchar3* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter )
{
	return launchResize<uchar3>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter);
}

// cudaResize (uchar4)
cudaError_t cudaResize( uchar4* input, size_t inputWidth, size_t inputHeight, uchar4* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter )
{
	return launchResize<uchar4>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter);
}

// cudaResize (float3)
cudaError_t cudaResize( float3* input, size_t inputWidth, size_t inputHeight, float3* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter )
{
	return launchResize<float3>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter);
}

// cudaResize (float4)
cudaError_t cudaResize( float4* input, size_t inputWidth, size_t inputHeight, float4* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter )
{
	return launchResize<float4>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter);
}

//-----------------------------------------------------------------------------------
cudaError_t cudaResize( void* input,  size_t inputWidth,  size_t inputHeight,
				    void* output, size_t outputWidth, size_t outputHeight, 
				    imageFormat format, cudaFilterMode filter )
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
		return cudaResize((uchar3*)input, inputWidth, inputHeight, (uchar3*)output, outputWidth, outputHeight, filter);
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return cudaResize((uchar4*)input, inputWidth, inputHeight, (uchar4*)output, outputWidth, outputHeight, filter);
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return cudaResize((float3*)input, inputWidth, inputHeight, (float3*)output, outputWidth, outputHeight, filter);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return cudaResize((float4*)input, inputWidth, inputHeight, (float4*)output, outputWidth, outputHeight, filter);
	else if( format == IMAGE_GRAY8 )
		return cudaResize((uint8_t*)input, inputWidth, inputHeight, (uint8_t*)output, outputWidth, outputHeight, filter);
	else if( format == IMAGE_GRAY32F )
		return cudaResize((float*)input, inputWidth, inputHeight, (float*)output, outputWidth, outputHeight, filter);

	printf("cudaResize() -- invalid image format %d\n", int(format));
	printf("                 supported formats are:\n");
	printf("                    * gray8\n");
	printf( "                    * gray32f\n");
	printf("                    * rgb8, bgr8\n");
	printf("                    * rgba8, bgra8\n");
	printf("                    * rgb32f, bgr32f\n");
	printf("                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}



