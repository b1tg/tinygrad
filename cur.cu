#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
typedef long unsigned int size_t;
#define half _Float16
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
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 1))) En26(float* data0_1, float* data1_1, float* data2_1, float* data3_1, float* data4_1, float* data5_1, float* data6_1, float* data7_1, float* data8_1, int* data9_1, int* data10_1, int* data11_1, int* data12_1, int* data13_1, int* data14_1, int* data15_1, int* data16_1, float* data17_1, float* data18_1, float* data19_1, float* data20_1, float* data21_1, float* data22_1, float* data23_1, float* data24_1) {
  int val0 = (*(data9_1+0));
  int val1 = (*(data10_1+0));
  int val2 = (*(data11_1+0));
  int val3 = (*(data12_1+0));
  int val4 = (*(data13_1+0));
  int val5 = (*(data14_1+0));
  int val6 = (*(data15_1+0));
  int val7 = (*(data16_1+0));
  float val8 = (*(data1_1+0));
  float val9 = (*(data2_1+0));
  float val10 = (*(data3_1+0));
  float val11 = (*(data4_1+0));
  float val12 = (*(data5_1+0));
  float val13 = (*(data6_1+0));
  float val14 = (*(data7_1+0));
  float val15 = (*(data8_1+0));
  float val16 = (*(data17_1+0));
  float val17 = (*(data18_1+0));
  float val18 = (*(data19_1+0));
  float val19 = (*(data20_1+0));
  float val20 = (*(data21_1+0));
  float val21 = (*(data22_1+0));
  float val22 = (*(data23_1+0));
  float val23 = (*(data24_1+0));
  *(data0_1+0) = (((((((((-val8-val9)-val10)-val11)-val12)-val13)-val14)-val15)*(1/((float)((((half)((val0+val1+val2+val3+val4+val5+val6+val7)))+((half)(1e-05f)))))))+((float)(((half)(((val16+val17+val18+val19+val20+val21+val22+val23)*0.00048828125f))))));
}