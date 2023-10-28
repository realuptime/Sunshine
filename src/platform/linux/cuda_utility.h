#ifndef __CUDA_UTILITY_H_
#define __CUDA_UTILITY_H_


#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/**
 * Execute a CUDA call and print out any errors
 * @return the original cudaError_t result
 * @ingroup cudaError
 */
#define CUDA(x)                         cudaCheckError((x), #x, __FILE__, __LINE__)

/**
 * Evaluates to true on success
 * @ingroup cudaError
 */
#define CUDA_SUCCESS(x)                 (CUDA(x) == cudaSuccess)

/**
 * Evaluates to true on failure
 * @ingroup cudaError
 */
#define CUDA_FAILED(x)                  (CUDA(x) != cudaSuccess)

/**
 * Return from the boolean function if CUDA call fails
 * @ingroup cudaError
 */
#define CUDA_VERIFY(x)                  if(CUDA_FAILED(x))      return false;

/**
 * Check for non-NULL pointer before freeing it, and then set the pointer to NULL.
 * @ingroup cudaError
 */
#define CUDA_FREE(x)            if(x != NULL) { cudaFree(x); x = NULL; }

/**
 * Check for non-NULL pointer before freeing it, and then set the pointer to NULL.
 * @ingroup cudaError
 */
#define CUDA_FREE_HOST(x)       if(x != NULL) { cudaFreeHost(x); x = NULL; }

/**
 * Check for non-NULL pointer before deleting it, and then set the pointer to NULL.
 * @ingroup util
 */
#define SAFE_DELETE(x)          if(x != NULL) { delete x; x = NULL; }

/**
 * Check for non-NULL pointer before freeing it, and then set the pointer to NULL.
 * @ingroup util
 */
#define SAFE_FREE(x)            if(x != NULL) { free(x); x = NULL; }

/**
 * If a / b has a remainder, round up.  This function is commonly using when launching
 * CUDA kernels, to compute a grid size inclusive of the entire dataset if it's dimensions
 * aren't evenly divisible by the block size.
 *
 * For example:
 *
 *    const dim3 blockDim(8,8);
 *    const dim3 gridDim(iDivUp(imgWidth,blockDim.x), iDivUp(imgHeight,blockDim.y));
 *
 * Then inside the CUDA kernel, there is typically a check that thread index is in-bounds.
 *
 * Without the use of iDivUp(), if the data dimensions weren't evenly divisible by the
 * block size, parts of the data wouldn't be covered by the grid and not processed.
 *
 * @ingroup cuda
 */
inline __device__ __host__ int iDivUp( int a, int b )           { return (a % b != 0) ? (a / b + 1) : (a / b); }

#endif
