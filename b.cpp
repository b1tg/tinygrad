#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
typedef unsigned char hip_fp8e5m2;
extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);
extern "C" __attribute__((device, const)) float __ocml_fmax_f32(float, float);
extern "C" __attribute__((device, pure)) float __ocml_exp2_f32(float);
extern "C" __attribute__((device, pure)) float __ocml_log2_f32(float);
extern "C" __attribute__((device, const)) float __ocml_sqrt_f32(float);
extern "C" __attribute__((device)) float __ocml_sin_f32(float);
extern "C" __attribute__((device)) float __ocml_trunc_f32(float);
extern "C" __attribute__((device, const)) double __ocml_fmax_f64(double, double);
extern "C" __attribute__((device, pure)) double __ocml_exp2_f64(double);
extern "C" __attribute__((device, pure)) double __ocml_log2_f64(double);
extern "C" __attribute__((device, const)) double __ocml_sqrt_f64(double);
extern "C" __attribute__((device)) double __ocml_sin_f64(double);
extern "C" __attribute__((device)) double __ocml_trunc_f64(double);
extern "C" __attribute__((device, const)) half __ocml_fmax_f16(half, half);
extern "C" __attribute__((device, pure)) half __ocml_exp2_f16(half);
extern "C" __attribute__((device, pure)) half __ocml_log2_f16(half);
extern "C" __attribute__((device, const)) half __ocml_sqrt_f16(half);
extern "C" __attribute__((device)) half __ocml_sin_f16(half);
extern "C" __attribute__((device)) half __ocml_trunc_f16(half);
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 32))) E_8_32_4(__hip_fp8_e5m2_fnuz* data0_1024, unsigned char* data1_1024) {
  int gidx0 = __ockl_get_group_id(0); /* 8 */
  int lidx0 = __ockl_get_local_id(0); /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  unsigned char val0 = (*(data1_1024+alu0));
  int alu1 = (alu0+1);
  unsigned char val1 = (*(data1_1024+alu1));
  int alu2 = (alu0+2);
  unsigned char val2 = (*(data1_1024+alu2));
  int alu3 = (alu0+3);
  unsigned char val3 = (*(data1_1024+alu3));
  *(data0_1024+alu1) = ((__hip_fp8_e5m2_fnuz)(val1));
  *(data0_1024+alu2) = ((__hip_fp8_e5m2_fnuz)(val2));
  *(data0_1024+alu3) = ((__hip_fp8_e5m2_fnuz)(val3));
  *(data0_1024+alu0) = ((__hip_fp8_e5m2_fnuz)(val0));
}