
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define hip_check(hip_call)                                                    \
{                                                                              \
    auto hip_res = hip_call;                                                   \
    if (hip_res != hipSuccess) {                                               \
      std::cerr << "Failed in hip call: " << #hip_call                         \
                << " with error: " << hipGetErrorName(hip_res) << std::endl;   \
      std::abort();                                                            \
    }                                                                          \
}

__device__ __hip_fp8_storage_t d_convert_float_to_fp8(
    float in, __hip_fp8_interpretation_t interpret, __hip_saturation_t sat) {
    return __hip_cvt_float_to_fp8(in, sat, interpret);
}

__device__ float d_convert_fp8_to_float(float in,
                                        __hip_fp8_interpretation_t interpret) {
    __half hf = __hip_cvt_fp8_to_halfraw(in, interpret);
    return hf;
}

__global__ void float_to_fp8_to_float(float *in,
                                    __hip_fp8_interpretation_t interpret,
                                    __hip_saturation_t sat, float *out,
                                    size_t size) {
    int i = threadIdx.x;
    if (i < size) {
        auto fp8 = d_convert_float_to_fp8(in[i], interpret, sat);
        out[i] = d_convert_fp8_to_float(fp8, interpret);
    }
}

__hip_fp8_storage_t
convert_float_to_fp8(float in, /* Input val */
                    __hip_fp8_interpretation_t
                        interpret, /* interpretation of number E4M3/E5M2 */
                    __hip_saturation_t sat /* Saturation behavior */
) {
    return __hip_cvt_float_to_fp8(in, sat, interpret);
}

float convert_fp8_to_float(
    __hip_fp8_storage_t in, /* Input val */
    __hip_fp8_interpretation_t
        interpret /* interpretation of number E4M3/E5M2 */
) {
    __half hf = __hip_cvt_fp8_to_halfraw(in, interpret);
    return hf;
}

int main() {
    constexpr size_t size = 32;
    hipDeviceProp_t prop;
    hip_check(hipGetDeviceProperties(&prop, 0));
    bool is_supported = (std::string(prop.gcnArchName).find("gfx94") != std::string::npos) || // gfx94x
                        (std::string(prop.gcnArchName).find("gfx120") != std::string::npos);  // gfx120x
    if(!is_supported) {
        std::cerr << "Need a gfx94x or gfx120x, but found: " << prop.gcnArchName << std::endl;
        std::cerr << "No device conversions are supported, only host conversions are supported." << std::endl;
        return -1;
    }

     __hip_fp8_interpretation_t interpret = (std::string(prop.gcnArchName).find("gfx94") != std::string::npos)
                                                    ? __HIP_E4M3_FNUZ // gfx94x
                                                    : __HIP_E4M3;     // gfx120x
    // interpret = __HIP_E4M3;
    // interpret
    constexpr __hip_saturation_t sat = __HIP_SATFINITE;

    std::vector<float> in;
    in.reserve(size);
    // for (size_t i = 0; i < size; i++) {
    //     in.push_back(i + 1.0f);
    // }
    // mi350
    // in.push_back(1);    // 0x38
    // in.push_back(3.25); // 0x45
    // in.push_back(251);  // 0x78

    in.push_back(1);    // 0x40
    in.push_back(3.25); // 0x4d
    in.push_back(251);
    // std::cout << "0x38 -> float: " << convert_fp8_to_float(0x38, interpret ) << "\n";
    // std::cout << "Converting float to fp8 and back..." << std::endl;
    // CPU convert
    std::vector<float> cpu_out;
    cpu_out.reserve(size);
    for (const auto &fval : in) {
        auto fp8 = convert_float_to_fp8(fval, __HIP_E4M3_FNUZ, sat);
        auto fp8_ocp = convert_float_to_fp8(fval, __HIP_E4M3, sat);
        // std::cout << "fp8: " << fp8 <<"\n";
        printf("float: %.3f, fnuz: %x, ocp: %x\n", fval, fp8, fp8_ocp);
        cpu_out.push_back(convert_fp8_to_float(fp8, interpret));
    }
    return 0;

    // GPU convert
    float *d_in, *d_out;
    hip_check(hipMalloc(&d_in, sizeof(float) * size));
    hip_check(hipMalloc(&d_out, sizeof(float) * size));

    hip_check(hipMemcpy(d_in, in.data(), sizeof(float) * in.size(),
                        hipMemcpyHostToDevice));

    float_to_fp8_to_float<<<1, size>>>(d_in, interpret, sat, d_out, size);

    std::vector<float> gpu_out(size, 0.0f);
    hip_check(hipMemcpy(gpu_out.data(), d_out, sizeof(float) * gpu_out.size(),
                        hipMemcpyDeviceToHost));

    hip_check(hipFree(d_in));
    hip_check(hipFree(d_out));

    // Validation
    for (size_t i = 0; i < size; i++) {
        if (cpu_out[i] != gpu_out[i]) {
            std::cerr << "cpu round trip result: " << cpu_out[i]
                      << " - gpu round trip result: " << gpu_out[i] << std::endl;
            std::abort();
        }
    }
    std::cout << "...CPU and GPU round trip convert matches." << std::endl;
}