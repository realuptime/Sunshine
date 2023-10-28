#include "nvenc_inc.h"

#include "nvenc_cuda.h"
#include "nvenc_utils.h"

#include "cuda_resize.h"

#if defined(_WIN32)
#define LOAD_FUNC(l, s) GetProcAddress(l, s)
#define DL_CLOSE_FUNC(l) FreeLibrary(l)
#else
#define LOAD_FUNC(l, s) dlsym(l, s)
#define DL_CLOSE_FUNC(l) dlclose(l)
#endif

// cuda_managed_memory.h
extern bool cudaAllocMapped(void** ptr, size_t width, size_t height, imageFormat format);
extern CUresult freeCudaCPU(void *ptr);
extern CUresult freeCudaGPU(void *ptr);

namespace cu
{

const char *GetBufferFormatName(const NV_ENC_BUFFER_FORMAT bufferFormat)
{
    switch (bufferFormat)
    {
    case NV_ENC_BUFFER_FORMAT_YV12: return "NV_ENC_BUFFER_FORMAT_YV12";
    case NV_ENC_BUFFER_FORMAT_IYUV: return "NV_ENC_BUFFER_FORMAT_IYUV";
    case NV_ENC_BUFFER_FORMAT_NV12: return "NV_ENC_BUFFER_FORMAT_NV12";
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT: return "NV_ENC_BUFFER_FORMAT_YUV420_10BIT";
    case NV_ENC_BUFFER_FORMAT_YUV444: return "NV_ENC_BUFFER_FORMAT_YUV444";
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT: return "NV_ENC_BUFFER_FORMAT_YUV444_10BIT";
    case NV_ENC_BUFFER_FORMAT_ARGB: return "NV_ENC_BUFFER_FORMAT_ARGB";
    case NV_ENC_BUFFER_FORMAT_ARGB10: return "NV_ENC_BUFFER_FORMAT_ARGB10";
    case NV_ENC_BUFFER_FORMAT_AYUV: return "NV_ENC_BUFFER_FORMAT_AYUV";
    case NV_ENC_BUFFER_FORMAT_ABGR: return "NV_ENC_BUFFER_FORMAT_ABGR";
    case NV_ENC_BUFFER_FORMAT_ABGR10: return "NV_ENC_BUFFER_FORMAT_ABGR10";
    default: return "Unknown";
    }
}


uint32_t GetChromaHeight(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t lumaHeight)
{
    switch (bufferFormat)
    {
    case NV_ENC_BUFFER_FORMAT_YV12:
    case NV_ENC_BUFFER_FORMAT_IYUV:
    case NV_ENC_BUFFER_FORMAT_NV12:
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
        return (lumaHeight + 1)/2;
    case NV_ENC_BUFFER_FORMAT_YUV444:
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
        return lumaHeight;
    case NV_ENC_BUFFER_FORMAT_ARGB:
    case NV_ENC_BUFFER_FORMAT_ARGB10:
    case NV_ENC_BUFFER_FORMAT_AYUV:
    case NV_ENC_BUFFER_FORMAT_ABGR:
    case NV_ENC_BUFFER_FORMAT_ABGR10:
        return 0;
    default:
        BOOST_LOG(error) << "Invalid Buffer format"; // NV_ENC_ERR_INVALID_PARAM;
        return 0;
    }
}

uint32_t GetWidthInBytes(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t width)
{
    switch (bufferFormat) {
    case NV_ENC_BUFFER_FORMAT_NV12:
    case NV_ENC_BUFFER_FORMAT_YV12:
    case NV_ENC_BUFFER_FORMAT_IYUV:
    case NV_ENC_BUFFER_FORMAT_YUV444:
        return width;
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
        return width * 2;
    case NV_ENC_BUFFER_FORMAT_ARGB:
    case NV_ENC_BUFFER_FORMAT_ARGB10:
    case NV_ENC_BUFFER_FORMAT_AYUV:
    case NV_ENC_BUFFER_FORMAT_ABGR:
    case NV_ENC_BUFFER_FORMAT_ABGR10:
        return width * 4;
    default:
        BOOST_LOG(error) << "Invalid Buffer format"; // NV_ENC_ERR_INVALID_PARAM);
        return 0;
    }
}

uint32_t GetNumChromaPlanes(const NV_ENC_BUFFER_FORMAT bufferFormat)
{
    switch (bufferFormat) 
    {
    case NV_ENC_BUFFER_FORMAT_NV12:
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
        return 1;
    case NV_ENC_BUFFER_FORMAT_YV12:
    case NV_ENC_BUFFER_FORMAT_IYUV:
    case NV_ENC_BUFFER_FORMAT_YUV444:
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
        return 2;
    case NV_ENC_BUFFER_FORMAT_ARGB:
    case NV_ENC_BUFFER_FORMAT_ARGB10:
    case NV_ENC_BUFFER_FORMAT_AYUV:
    case NV_ENC_BUFFER_FORMAT_ABGR:
    case NV_ENC_BUFFER_FORMAT_ABGR10:
        return 0;
    default:
        BOOST_LOG(error) << "Invalid Buffer format"; // NV_ENC_ERR_INVALID_PARAM);
        return -1;
    }
}

} // namespace cu 

namespace nvenc {

  void
  cff(CudaFunctions *cf) {
    cuda_free_functions(&cf);
  }
  using cdf_t = util::safe_ptr<CudaFunctions, cff>;
  static bool cdfInitialized = false;
  static cdf_t cdf;

  inline static int
  check(CUresult result, const std::string_view &sv) {
    if (result != CUDA_SUCCESS)
    {
      const char *name;
      const char *description;

      cdf->cuGetErrorName(result, &name);
      cdf->cuGetErrorString(result, &description);

      BOOST_LOG(error) << sv << ' ' << name << ": " << description;
      return false;
    }
    return true;
  }

    CUresult freeDevicePtr(void *ptr)
    {
#if 1
        CUresult ret = freeCudaCPU(ptr);
        if (!check(ret, "CPU free"))
            return ret;
        //ret = freeCudaGPU(ptr);
        //if (!check(ret, "GPU free"))
        //    return ret;
#else
        CUresult ret = cdf->cuMemFree((CUdeviceptr)ptr);
#endif
        return ret;
    }

  CUcontext init_cuda()
  {
    auto status = cuda_load_functions(&cdf, nullptr);
    if (status) {
      BOOST_LOG(error) << "Couldn't load cuda: " << status;

      return 0;
    }

    if (!check(cdf->cuInit(0), "Couldn't initialize cuda"))
    {
        return 0;
    }

    CUdevice dev;
    if (!check(cdf->cuDeviceGet(&dev, 0), "cuDeviceGet failed"))
    {
        return 0;
    }
    BOOST_LOG(info) << "CUDA dev:" << dev;

    CUcontext cu_context = 0;
    if (!check(cdf->cuCtxCreate(&cu_context, 0, dev), "cuCtxCreate failed"))
    {
        return 0;
    }
    BOOST_LOG(info) << "CUDA context:" << cu_context;

    return cu_context;
  }

  bool fini_cuda(CUcontext context)
  {
      if (!check(cdf->cuCtxDestroy(context), "cuCtxDestroy failed!"))
      {
          return false;
      }
      return true;
  }

  nvenc_cuda::nvenc_cuda(CUcontext cu_context):
      nvenc_base(NV_ENC_DEVICE_TYPE_CUDA, cu_context),
      cuda_context(cu_context),
      encoder_deviceptr(nullptr),
      input_deviceptr(nullptr),
      libHandle(nullptr),
      encoderDevicePtrPitch(0),
      cudaWidth(0),
      cudaHeight(0),
      cuda_array(nullptr),
      stream(nullptr)
  {
  }

  void nvenc_cuda::closeLibrary()
  {
    if (libHandle)
    {
        DL_CLOSE_FUNC(libHandle);
        libHandle = NULL;
    }
  }

  nvenc_cuda::~nvenc_cuda()
  {
      BOOST_LOG(info) << "nvenc::~nvenc()";
    if (encoder) destroy_encoder();
    
    stream.reset();
    if (cuda_array)
    {
            cdf->cuArrayDestroy(cuda_array);
            cuda_array = nullptr;
    }

    if (encoder_deviceptr)
    {
        check(freeDevicePtr(encoder_deviceptr), "free encoder_deviceptr failed");
        encoder_deviceptr = nullptr;
    }

    if (input_deviceptr)
    {
        check(freeDevicePtr(input_deviceptr), "free input_deviceptr failed");
        input_deviceptr = nullptr;
    }

    check(cdf->cuCtxDestroy(cuda_context), "cuCtxDestroy failed!");

    closeLibrary();
  }

  bool
  nvenc_cuda::init_library()
  {
    if (libHandle) return true;

#ifdef _WIN32
  #ifdef _WIN64
    auto libName = "nvEncodeAPI64.dll";
  #else
    auto libName = "nvEncodeAPI.dll";
  #endif
    libHandle = LoadLibraryEx(libName, NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
    auto libName = "libnvidia-encode.so.1";
    libHandle = dlopen(libName, RTLD_LAZY | RTLD_LOCAL);
#endif

    if (libHandle == nullptr)
    {
        BOOST_LOG(debug) << "Couldn't load NvEnc library " << libName;
        return false;
    }

    auto create_instance = (decltype(NvEncodeAPICreateInstance) *) LOAD_FUNC(libHandle, "NvEncodeAPICreateInstance");
    if (!create_instance)
    {
      BOOST_LOG(error) << "No NvEncodeAPICreateInstance in " << libName;
      closeLibrary();
      return false;
    }

    auto new_nvenc = std::make_unique<NV_ENCODE_API_FUNCTION_LIST>();
    memset(new_nvenc.get(), 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    new_nvenc->version = NV_ENCODE_API_FUNCTION_LIST_VER;
    if (nvenc_failed(create_instance(new_nvenc.get())))
    {
      BOOST_LOG(error) << "NvEncodeAPICreateInstance failed: " << last_error_string;
      closeLibrary();
      return false;
    }

    if (!cdfInitialized)
    {
        auto status = cuda_load_functions(&cdf, nullptr);
        if (status) {
          BOOST_LOG(error) << "Couldn't load cuda: " << status;
          return false;
        }
        cdfInitialized = true;
    }

    BOOST_LOG(info) << "nvenc_cuda initialized!";

    nvenc = std::move(new_nvenc);
    return true;
  }

  bool nvenc_cuda::transferToDevice(platf::img_t &img)
  {
        //return true;

        if (!input_deviceptr || img.width != cudaWidth || img.height != cudaHeight)
        {
            if (input_deviceptr)
            {
                check(freeDevicePtr(input_deviceptr), "free input_deviceptr failed");
                input_deviceptr = nullptr;
            }
            cdf->cuCtxPushCurrent(cuda_context);
            if (!cudaAllocMapped(&input_deviceptr, img.width, img.height, IMAGE_RGBA8))
            {
                BOOST_LOG(error) << "cudaAllocMapped failed";
                input_deviceptr = nullptr;
                cdf->cuCtxPopCurrent(NULL);
                return false;
            }
            BOOST_LOG(info) << "cudaAllocMapped succeeded! input_deviceptr:" << input_deviceptr << " res:" << img.width << "x" << img.height;
            cdf->cuCtxPopCurrent(NULL);

            if (input_deviceptr)
            {
                cudaWidth = img.width;
                cudaHeight = img.height;
            }
        }

        cdf->cuCtxPushCurrent(cuda_context);
        // copy to GPU
        memcpy(input_deviceptr, img.data, img.row_pitch * img.height);
        // Copy input to encoder deviceptr
        auto resizeRet = cudaResize(input_deviceptr, img.width, img.height,
            encoder_deviceptr, encoder_params.width, encoder_params.height,
            IMAGE_RGBA8, FILTER_LINEAR);
        if (!check(resizeRet, "cudaResize failed"))
        {
            cdf->cuCtxPopCurrent(NULL);
            return false;
        }
        cdf->cuCtxPopCurrent(NULL);

        bool ret = true;
#if 0
        cdf->cuCtxPushCurrent(cuda_context);

        CUDA_MEMCPY2D param = { 0, };
        param.srcMemoryType = CU_MEMORYTYPE_HOST;
        param.srcHost = img.data;
        param.srcPitch = img.row_pitch;
        param.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        param.dstDevice = input_deviceptr;
        param.dstPitch = encoderDevicePtrPitch;
        //param.WidthInBytes = cu::GetWidthInBytes(encoder_params.buffer_format, img.width);
        param.WidthInBytes = img.row_pitch;
        param.Height = img.height;
        bool cudaRes = false;
        if (stream)
        {
            cudaRes = check(cdf->cuMemcpy2DAsync(&param, stream.get()), "cuMemcpy2DAsync failed");
        }
        else
        {
            cudaRes = check(cdf->cuMemcpy2D(&param), "cuMemcpy2D failed");
        }
        if (!cudaRes)
        {
            BOOST_LOG(error) << "CUDA: cuMemcpy2DAsync failed! "
                << " enc_res:" << encoder_params.width << "x" << encoder_params.height
                << " img_res:" << img.width << "x" << img.height
                << " ptr:" << input_deviceptr
                << " img.row_pitch:" << img.row_pitch
                << " WidthInBytes:" << param.WidthInBytes;
            ret = false;
        }

        if (stream)
        {
            check(cdf->cuStreamSynchronize(stream.get()), "CuStreamSynchronize failed");
        }
        cdf->cuCtxPopCurrent(NULL);
#endif
        //BOOST_LOG(info) << "transferToDevice: ret:" << ret;
        test = 1;

        return ret;
  }

  bool nvenc_cuda::register_deviceptr(void *devicePtr) // CUdeviceptr
  {
      unregister_resource();

      NV_ENC_REGISTER_RESOURCE register_resource = { 0 };
      register_resource.version = NV_ENC_REGISTER_RESOURCE_VER;
      register_resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
      register_resource.width = encoder_params.width;
      register_resource.height = encoder_params.height;
      // TODO: from format
      //register_resource.pitch = encoder_params.width * 4;
      register_resource.pitch = encoderDevicePtrPitch;
      register_resource.bufferFormat = encoder_params.buffer_format;
      register_resource.bufferUsage = NV_ENC_INPUT_IMAGE;
      register_resource.resourceToRegister = devicePtr;

      if (nvenc_failed(nvenc->nvEncRegisterResource(encoder, &register_resource)))
      {
        BOOST_LOG(error) << "NvEncRegisterResource failed: " << last_error_string;
        return false;
      }
      BOOST_LOG(info) << "NvEncRegisterResource succeded! ptr:" << (CUdeviceptr)devicePtr;

      registered_input_buffer = register_resource.registeredResource;

      return true;
  }

  CUdeviceptr nvenc_cuda::alloc_pitched(uint32_t width, uint32_t height, size_t &pitch, const char *msg)
  {
        const auto fmt = encoder_params.buffer_format;
        const auto w = width, h = height;

        CUdeviceptr ret = 0;

        uint32_t chromaHeight = cu::GetNumChromaPlanes(fmt) * cu::GetChromaHeight(fmt, h);
        if (fmt == NV_ENC_BUFFER_FORMAT_YV12 || fmt == NV_ENC_BUFFER_FORMAT_IYUV)
            chromaHeight = cu::GetChromaHeight(fmt, h);

        if (!check(cdf->cuMemAllocPitch(
            &ret,
            &pitch,
            cu::GetWidthInBytes(fmt, w),
            h + chromaHeight,
            16), "cuMemAllocPitch failed"))
        {
            ret = 0;
        }
        else
        {
            BOOST_LOG(info)
                << "CUDA: cuMemAllocPitch(" << msg << ") succeeded."
                << " pitch:" << pitch
                << " deviceptr:" << ret 
                << " res:" << width << "x" << height 
                << " fmt:" << cu::GetBufferFormatName(encoder_params.buffer_format);
        }

        return ret;
  }

  bool
  nvenc_cuda::create_and_register_input_buffer()
  {
    if (!encoder_deviceptr)
    {
        cdf->cuCtxPushCurrent(cuda_context);
        #if 0
        encoder_deviceptr = alloc_pitched(encoder_params.width, encoder_params.height, encoderDevicePtrPitch, "encoder buffer");
        #else
        if (!cudaAllocMapped(&encoder_deviceptr, encoder_params.width, encoder_params.height, IMAGE_RGBA8))
        {
            BOOST_LOG(error) << "cudaAllocMapped failed (encoder_deviceptr)";
            encoder_deviceptr = nullptr;
        }
        else
        {
            encoderDevicePtrPitch = encoder_params.width * 4;
        }
        #endif
        cdf->cuCtxPopCurrent(NULL);
    }

    if (!registered_input_buffer && encoder_deviceptr)
    {
        cdf->cuCtxPushCurrent(cuda_context);
        register_deviceptr(encoder_deviceptr);
        cdf->cuCtxPopCurrent(NULL);
    }

/*
    stream = cuda::make_stream();
    auto pstream = stream.get();
    if (nvenc_failed(nvenc->nvEncSetIOCudaStreams(encoder, &pstream, &pstream))) {
      BOOST_LOG(error) << "NvEncSetIOCudaStreams failed: " << last_error_string;
      return false;
    }
*/
    return true;
  }

  bool nvenc_cuda::unregister_resource()
  {
      if (!registered_input_buffer)
          return true;

      if (nvenc_failed(nvenc->nvEncUnregisterResource(encoder, registered_input_buffer)))
      {
        BOOST_LOG(error) << "NvEncRegisterResource failed: " << last_error_string;
        return false;
      }

      registered_input_buffer = nullptr;

      return true;
  }

}  // namespace nvenc

