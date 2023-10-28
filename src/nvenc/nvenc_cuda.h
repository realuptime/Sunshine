#pragma once

#if 1

#include "nvenc_base.h"
#include "src/platform/linux/cuda.h"

struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
typedef unsigned long long CUdeviceptr;

namespace nvenc
{
  CUcontext init_cuda();
  bool fini_cuda(CUcontext context);

  class nvenc_cuda final: public nvenc_base {
  public:
    nvenc_cuda(CUcontext cu_context);
    ~nvenc_cuda();

  bool transferToDevice(platf::img_t &img);
  bool register_deviceptr(void *devicePtr); // CUdeviceptr
  bool register_cudaarray(uint32_t width, uint32_t height);
  bool unregister_resource();

  private:
    bool
    init_library() override;

    bool
    create_and_register_input_buffer() override;

    void closeLibrary();

    CUcontext cuda_context;
    CUdeviceptr cuda_deviceptr;
    void *libHandle;
    size_t cudaPitch;

    uint32_t cudaWidth, cudaHeight;

    CUarray cuda_array;

    cuda::stream_t stream;

    cuda::sws_t sws;
    cuda::tex_t tex;
  };

}  // namespace nvenc
#endif

