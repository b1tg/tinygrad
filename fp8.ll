
; __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx942
; ModuleID = 'fp8.cpp'
source_filename = "fp8.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__hip_cuid_ffca4d332f194385 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_ffca4d332f194385 to ptr)], section "llvm.metadata"

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #0

; Function Attrs: mustprogress norecurse nounwind memory(argmem: readwrite, inaccessiblemem: write)
define protected amdgpu_kernel void @_Z21float_to_fp8_to_floatPf26__hip_fp8_interpretation_t18__hip_saturation_tS_m(ptr addrspace(1) nocapture noundef readonly %0, i32 noundef %1, i32 noundef %2, ptr addrspace(1) nocapture noundef writeonly %3, i64 noundef %4) local_unnamed_addr #1 {
  %6 = tail call noundef i32 @llvm.amdgcn.workitem.id.x(), !range !6, !noundef !7
  %7 = zext nneg i32 %6 to i64
  %8 = icmp ult i64 %7, %4
  br i1 %8, label %9, label %90

9:                                                ; preds = %5
  %10 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %7
  %11 = load float, ptr addrspace(1) %10, align 4, !tbaa !8, !amdgpu.noclobber !7
  %12 = add i32 %1, -4
  %13 = icmp ult i32 %12, -2
  br i1 %13, label %14, label %15

14:                                               ; preds = %9
  tail call void @llvm.trap()
  unreachable

15:                                               ; preds = %9
  %16 = icmp eq i32 %2, 1
  br i1 %16, label %17, label %28

17:                                               ; preds = %15
  %18 = bitcast float %11 to i32
  %19 = and i32 %18, 2139095040
  %20 = icmp eq i32 %19, 2139095040
  %21 = icmp eq i32 %1, 2
  br i1 %21, label %22, label %25

22:                                               ; preds = %17
  %23 = tail call contract float @llvm.amdgcn.fmed3.f32(float %11, float 2.400000e+02, float -2.400000e+02)
  %24 = select i1 %20, float %11, float %23
  br label %28

25:                                               ; preds = %17
  %26 = tail call contract float @llvm.amdgcn.fmed3.f32(float %11, float 5.734400e+04, float -5.734400e+04)
  %27 = select i1 %20, float %11, float %26
  br label %28

28:                                               ; preds = %25, %22, %15
  %29 = phi float [ %11, %15 ], [ %24, %22 ], [ %27, %25 ]
  %30 = and i32 %1, 1
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %32, label %34

32:                                               ; preds = %28
  %33 = tail call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %29, float %29, i32 0, i1 false)
  br label %36

34:                                               ; preds = %28
  %35 = tail call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %29, float %29, i32 0, i1 false)
  br label %36

36:                                               ; preds = %32, %34
  %37 = phi i32 [ %33, %32 ], [ %35, %34 ]
  %38 = trunc i32 %37 to i8
  %39 = icmp eq i32 %1, 2
  %40 = select i1 %39, i32 3, i32 2
  %41 = icmp eq i8 %38, 0
  br i1 %41, label %87, label %42

42:                                               ; preds = %36
  %43 = shl nsw i32 -1, %40
  %44 = xor i32 %43, -1
  %45 = and i32 %37, %44
  %46 = zext nneg i32 %45 to i64
  %47 = icmp eq i8 %38, -128
  br i1 %47, label %87, label %48

48:                                               ; preds = %42
  %49 = and i32 %37, 127
  %50 = lshr i32 %49, %40
  %51 = icmp eq i32 %50, 0
  br i1 %51, label %52, label %63

52:                                               ; preds = %48
  %53 = tail call noundef i32 @llvm.ctlz.i32(i32 %45, i1 false), !range !12
  %54 = add nuw nsw i32 %53, %40
  %55 = add nsw i32 %54, -31
  %56 = zext nneg i32 %55 to i64
  %57 = shl nuw nsw i64 %46, %56
  %58 = sub nsw i32 32, %54
  %59 = zext nneg i32 %40 to i64
  %60 = shl nsw i64 -1, %59
  %61 = xor i64 %60, -1
  %62 = and i64 %57, %61
  br label %63

63:                                               ; preds = %52, %48
  %64 = phi i32 [ %58, %52 ], [ %50, %48 ]
  %65 = phi i64 [ %62, %52 ], [ %46, %48 ]
  %66 = select i1 %39, i32 7, i32 -1
  %67 = add nsw i32 %64, %66
  %68 = sub nuw nsw i32 10, %40
  %69 = zext nneg i32 %68 to i64
  %70 = shl nuw nsw i64 %65, %69
  %71 = icmp slt i32 %67, 1
  %72 = or i64 %70, 1024
  %73 = sub nsw i32 1, %67
  %74 = zext nneg i32 %73 to i64
  %75 = lshr i64 %72, %74
  %76 = shl nuw nsw i32 %67, 10
  %77 = select i1 %71, i64 %75, i64 %70
  %78 = shl i32 %37, 8
  %79 = and i32 %78, 32768
  %80 = select i1 %71, i32 0, i32 %76
  %81 = or i32 %80, %79
  %82 = zext nneg i32 %81 to i64
  %83 = or i64 %77, %82
  %84 = trunc i64 %83 to i16
  %85 = bitcast i16 %84 to half
  %86 = fpext half %85 to float
  br label %87

87:                                               ; preds = %36, %42, %63
  %88 = phi float [ 0.000000e+00, %36 ], [ %86, %63 ], [ 0x7FF8040000000000, %42 ]
  %89 = getelementptr inbounds float, ptr addrspace(1) %3, i64 %7
  store float %88, ptr addrspace(1) %89, align 4, !tbaa !8
  br label %90

90:                                               ; preds = %87, %5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.fmed3.f32(float, float, float) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.cvt.pk.bf8.f32(float, float, i32, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctlz.i32(i32, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #1 = { mustprogress norecurse nounwind memory(argmem: readwrite, inaccessiblemem: write) "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="true" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"amdgpu_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 2, i32 0}
!5 = !{!"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.3 25012 e5bf7e55c91490b07c49d8960fa7983d864936c4)"}
!6 = !{i32 0, i32 1024}
!7 = !{}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{i32 0, i32 33}

; __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx942

; __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
; ModuleID = 'fp8.cpp'
source_filename = "fp8.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%struct.dim3 = type { i32, i32, i32 }
%struct.hipDeviceProp_tR0600 = type { [256 x i8], %struct.hipUUID_t, [8 x i8], i32, i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [63 x i32], [32 x i32], [256 x i8], i64, i32, %struct.hipDeviceArch_t, ptr, ptr, i32, i32, i32, i32, i32, i32 }
%struct.hipUUID_t = type { [16 x i8] }
%struct.hipDeviceArch_t = type { i24 }
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char>::_Alloc_hider", i64, %union.anon.1 }
%"struct.std::__cxx11::basic_string<char>::_Alloc_hider" = type { ptr }
%union.anon.1 = type { i64, [8 x i8] }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], ptr, i8, [7 x i8], ptr, ptr, ptr, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ ptr, i32 }>

@_Z21float_to_fp8_to_floatPf26__hip_fp8_interpretation_t18__hip_saturation_tS_m = dso_local constant ptr @_Z36__device_stub__float_to_fp8_to_floatPf26__hip_fp8_interpretation_t18__hip_saturation_tS_m, align 8
@_ZSt4cerr = external dso_local global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [21 x i8] c"Failed in hip call: \00", align 1
@.str.1 = private unnamed_addr constant [33 x i8] c"hipGetDeviceProperties(&prop, 0)\00", align 1
@.str.2 = private unnamed_addr constant [14 x i8] c" with error: \00", align 1
@.str.3 = private unnamed_addr constant [6 x i8] c"gfx94\00", align 1
@.str.4 = private unnamed_addr constant [7 x i8] c"gfx120\00", align 1
@.str.5 = private unnamed_addr constant [38 x i8] c"Need a gfx94x or gfx120x, but found: \00", align 1
@.str.6 = private unnamed_addr constant [74 x i8] c"No device conversions are supported, only host conversions are supported.\00", align 1
@_ZSt4cout = external dso_local global %"class.std::basic_ostream", align 8
@.str.7 = private unnamed_addr constant [16 x i8] c"0x38 -> float: \00", align 1
@.str.8 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.9 = private unnamed_addr constant [36 x i8] c"Converting float to fp8 and back...\00", align 1
@.str.10 = private unnamed_addr constant [6 x i8] c"fp8: \00", align 1
@.str.11 = private unnamed_addr constant [39 x i8] c"hipMalloc(&d_in, sizeof(float) * size)\00", align 1
@.str.12 = private unnamed_addr constant [40 x i8] c"hipMalloc(&d_out, sizeof(float) * size)\00", align 1
@.str.13 = private unnamed_addr constant [77 x i8] c"hipMemcpy(d_in, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice)\00", align 1
@.str.14 = private unnamed_addr constant [88 x i8] c"hipMemcpy(gpu_out.data(), d_out, sizeof(float) * gpu_out.size(), hipMemcpyDeviceToHost)\00", align 1
@.str.15 = private unnamed_addr constant [14 x i8] c"hipFree(d_in)\00", align 1
@.str.16 = private unnamed_addr constant [15 x i8] c"hipFree(d_out)\00", align 1
@.str.17 = private unnamed_addr constant [24 x i8] c"cpu round trip result: \00", align 1
@.str.18 = private unnamed_addr constant [27 x i8] c" - gpu round trip result: \00", align 1
@.str.19 = private unnamed_addr constant [43 x i8] c"...CPU and GPU round trip convert matches.\00", align 1
@0 = private unnamed_addr constant [79 x i8] c"_Z21float_to_fp8_to_floatPf26__hip_fp8_interpretation_t18__hip_saturation_tS_m\00", align 1
@__hip_fatbin_ffca4d332f194385 = external constant i8, section ".hip_fatbin"
@__hip_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1212764230, i32 1, ptr @__hip_fatbin_ffca4d332f194385, ptr null }, section ".hipFatBinSegment", align 8
@__hip_gpubin_handle_ffca4d332f194385 = internal unnamed_addr global ptr null, align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__hip_module_ctor, ptr null }]
@__hip_cuid_ffca4d332f194385 = global i8 0
@llvm.compiler.used = appending global [1 x ptr] [ptr @__hip_cuid_ffca4d332f194385], section "llvm.metadata"

; Function Attrs: mustprogress norecurse uwtable
define dso_local void @_Z36__device_stub__float_to_fp8_to_floatPf26__hip_fp8_interpretation_t18__hip_saturation_tS_m(ptr noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %3, i64 noundef %4) #0 {
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca ptr, align 8
  %10 = alloca i64, align 8
  %11 = alloca %struct.dim3, align 8
  %12 = alloca %struct.dim3, align 8
  %13 = alloca i64, align 8
  %14 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8, !tbaa !3
  store i32 %1, ptr %7, align 4, !tbaa !7
  store i32 %2, ptr %8, align 4, !tbaa !9
  store ptr %3, ptr %9, align 8, !tbaa !3
  store i64 %4, ptr %10, align 8, !tbaa !11
  %15 = alloca [5 x ptr], align 16
  store ptr %6, ptr %15, align 16
  %16 = getelementptr inbounds ptr, ptr %15, i64 1
  store ptr %7, ptr %16, align 8
  %17 = getelementptr inbounds ptr, ptr %15, i64 2
  store ptr %8, ptr %17, align 16
  %18 = getelementptr inbounds ptr, ptr %15, i64 3
  store ptr %9, ptr %18, align 8
  %19 = getelementptr inbounds ptr, ptr %15, i64 4
  store ptr %10, ptr %19, align 16
  %20 = call i32 @__hipPopCallConfiguration(ptr nonnull %11, ptr nonnull %12, ptr nonnull %13, ptr nonnull %14)
  %21 = load i64, ptr %13, align 8
  %22 = load ptr, ptr %14, align 8
  %23 = load i64, ptr %11, align 8
  %24 = getelementptr inbounds i8, ptr %11, i64 8
  %25 = load i32, ptr %24, align 8
  %26 = load i64, ptr %12, align 8
  %27 = getelementptr inbounds i8, ptr %12, i64 8
  %28 = load i32, ptr %27, align 8
  %29 = call noundef i32 @hipLaunchKernel(ptr noundef nonnull @_Z21float_to_fp8_to_floatPf26__hip_fp8_interpretation_t18__hip_saturation_tS_m, i64 %23, i32 %25, i64 %26, i32 %28, ptr noundef nonnull %15, i64 noundef %21, ptr noundef %22)
  ret void
}

declare dso_local i32 @__hipPopCallConfiguration(ptr, ptr, ptr, ptr) local_unnamed_addr

declare dso_local i32 @hipLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef zeroext i8 @_Z20convert_float_to_fp8f26__hip_fp8_interpretation_t18__hip_saturation_t(float noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #2 {
  %4 = and i32 %1, -2
  %5 = icmp eq i32 %4, 2
  %6 = icmp eq i32 %2, 1
  %7 = bitcast float %0 to i32
  %8 = zext i32 %7 to i64
  %9 = and i64 %8, 8388607
  %10 = lshr i32 %7, 23
  %11 = and i32 %10, 255
  %12 = lshr i32 %7, 24
  %13 = and i32 %12, 128
  br i1 %5, label %14, label %124

14:                                               ; preds = %3
  %15 = icmp eq i32 %1, 2
  %16 = select i1 %15, i32 4, i32 5
  %17 = select i1 %15, i32 3, i32 2
  %18 = or i32 %12, 127
  %19 = select i1 %6, i32 %18, i32 128
  %20 = and i64 %8, 2139095040
  %21 = icmp eq i64 %20, 2139095040
  br i1 %21, label %22, label %24

22:                                               ; preds = %14
  %23 = trunc i32 %19 to i8
  br label %244

24:                                               ; preds = %14
  %25 = select i1 %15, i64 1131413504, i64 1197473792
  %26 = and i64 %8, 2147483647
  %27 = icmp ugt i64 %26, %25
  br i1 %27, label %28, label %30

28:                                               ; preds = %24
  %29 = trunc i32 %19 to i8
  br label %244

30:                                               ; preds = %24
  %31 = icmp eq i32 %7, 0
  br i1 %31, label %244, label %32

32:                                               ; preds = %30
  %33 = add nsw i32 %16, -1
  %34 = shl nuw nsw i32 1, %33
  %35 = icmp eq i32 %11, 0
  br i1 %35, label %36, label %38

36:                                               ; preds = %32
  %37 = sub nuw nsw i32 127, %34
  br label %45

38:                                               ; preds = %32
  %39 = sub nsw i32 1, %34
  %40 = add nsw i32 %11, -127
  %41 = icmp sgt i32 %40, %39
  %42 = sub nsw i32 %39, %40
  %43 = select i1 %41, i32 0, i32 %42
  %44 = or disjoint i64 %9, 8388608
  br label %45

45:                                               ; preds = %38, %36
  %46 = phi i32 [ -126, %36 ], [ %40, %38 ]
  %47 = phi i32 [ %37, %36 ], [ %43, %38 ]
  %48 = phi i64 [ %9, %36 ], [ %44, %38 ]
  %49 = sub nuw nsw i32 23, %17
  %50 = add nsw i32 %47, %49
  %51 = zext nneg i32 %50 to i64
  %52 = shl nsw i64 -1, %51
  %53 = xor i64 %52, -1
  %54 = and i64 %48, %53
  %55 = add nsw i32 %50, -1
  %56 = zext nneg i32 %55 to i64
  %57 = shl nuw i64 1, %56
  %58 = icmp eq i64 %54, %57
  %59 = icmp sgt i32 %47, 0
  %60 = zext nneg i32 %47 to i64
  %61 = lshr i64 %48, %60
  %62 = icmp eq i32 %47, -1
  %63 = zext i1 %62 to i64
  %64 = shl nuw nsw i64 %48, %63
  %65 = select i1 %59, i64 %61, i64 %64
  %66 = add nsw i32 %46, %34
  %67 = add nsw i32 %66, %47
  %68 = trunc i64 %65 to i32
  %69 = lshr i32 %68, 23
  %70 = and i32 %69, 1
  %71 = add nsw i32 %67, %70
  %72 = add nsw i32 %71, -1
  %73 = zext nneg i32 %49 to i64
  %74 = shl nuw nsw i64 1, %73
  %75 = add nsw i64 %74, -1
  %76 = and i64 %65, %74
  %77 = icmp eq i64 %76, 0
  %78 = select i1 %58, i1 %77, i1 false
  %79 = sext i1 %78 to i64
  %80 = add nsw i64 %65, %79
  %81 = and i64 %80, %75
  %82 = add nuw nsw i64 %81, %65
  %83 = icmp eq i32 %72, 0
  br i1 %83, label %84, label %88

84:                                               ; preds = %45
  %85 = trunc i64 %82 to i32
  %86 = lshr i32 %85, 23
  %87 = and i32 %86, 1
  br label %94

88:                                               ; preds = %45
  %89 = and i64 %82, 16777216
  %90 = icmp eq i64 %89, 0
  %91 = select i1 %90, i32 %72, i32 %71
  %92 = lshr exact i64 %89, 24
  %93 = lshr i64 %82, %92
  br label %94

94:                                               ; preds = %88, %84
  %95 = phi i32 [ %87, %84 ], [ %91, %88 ]
  %96 = phi i64 [ %82, %84 ], [ %93, %88 ]
  %97 = shl nsw i32 -1, %16
  %98 = xor i32 %97, -1
  %99 = icmp sgt i32 %95, %98
  br i1 %99, label %100, label %105

100:                                              ; preds = %94
  br i1 %6, label %101, label %244

101:                                              ; preds = %100
  %102 = shl nsw i32 -1, %17
  %103 = xor i32 %102, -1
  %104 = zext nneg i32 %103 to i64
  br label %114

105:                                              ; preds = %94
  %106 = lshr i64 %96, %73
  %107 = icmp eq i32 %95, 0
  %108 = icmp eq i64 %106, 0
  %109 = select i1 %107, i1 %108, i1 false
  br i1 %109, label %244, label %110

110:                                              ; preds = %105
  %111 = shl nsw i32 -1, %17
  %112 = xor i32 %111, -1
  %113 = zext nneg i32 %112 to i64
  br label %114

114:                                              ; preds = %110, %101
  %115 = phi i64 [ %113, %110 ], [ %104, %101 ]
  %116 = phi i64 [ %106, %110 ], [ %104, %101 ]
  %117 = phi i32 [ %95, %110 ], [ %98, %101 ]
  %118 = and i64 %116, %115
  %119 = shl nsw i32 %117, %17
  %120 = or i32 %119, %13
  %121 = zext i32 %120 to i64
  %122 = or i64 %118, %121
  %123 = trunc i64 %122 to i8
  br label %244

124:                                              ; preds = %3
  %125 = icmp eq i32 %1, 0
  %126 = select i1 %125, i32 4, i32 5
  %127 = select i1 %125, i32 3, i32 2
  %128 = select i1 %6, i32 123, i32 124
  %129 = select i1 %6, i32 126, i32 127
  %130 = select i1 %125, i32 %129, i32 %128
  %131 = or disjoint i32 %130, %13
  %132 = and i64 %8, 2139095040
  %133 = icmp eq i64 %132, 2139095040
  br i1 %133, label %134, label %139

134:                                              ; preds = %124
  %135 = or i32 %12, 127
  %136 = icmp eq i64 %9, 0
  %137 = select i1 %136, i32 %131, i32 %135
  %138 = trunc i32 %137 to i8
  br label %244

139:                                              ; preds = %124
  %140 = select i1 %125, i64 1138753536, i64 1197473792
  %141 = and i64 %8, 2147483647
  %142 = icmp ugt i64 %141, %140
  br i1 %142, label %143, label %145

143:                                              ; preds = %139
  %144 = trunc i32 %131 to i8
  br label %244

145:                                              ; preds = %139
  %146 = icmp eq i32 %7, 0
  br i1 %146, label %244, label %147

147:                                              ; preds = %145
  %148 = add nsw i32 %126, -1
  %149 = shl nsw i32 -1, %148
  %150 = xor i32 %149, -1
  %151 = icmp eq i32 %11, 0
  br i1 %151, label %152, label %154

152:                                              ; preds = %147
  %153 = add nsw i32 %149, 128
  br label %161

154:                                              ; preds = %147
  %155 = add nuw nsw i32 %149, 2
  %156 = add nsw i32 %11, -127
  %157 = icmp sgt i32 %156, %155
  %158 = sub nsw i32 %155, %156
  %159 = select i1 %157, i32 0, i32 %158
  %160 = or disjoint i64 %9, 8388608
  br label %161

161:                                              ; preds = %154, %152
  %162 = phi i32 [ -126, %152 ], [ %156, %154 ]
  %163 = phi i32 [ %153, %152 ], [ %159, %154 ]
  %164 = phi i64 [ %9, %152 ], [ %160, %154 ]
  %165 = sub nuw nsw i32 23, %127
  %166 = add nsw i32 %163, %165
  %167 = zext nneg i32 %166 to i64
  %168 = shl nsw i64 -1, %167
  %169 = xor i64 %168, -1
  %170 = and i64 %164, %169
  %171 = add nsw i32 %166, -1
  %172 = zext nneg i32 %171 to i64
  %173 = shl nuw i64 1, %172
  %174 = icmp eq i64 %170, %173
  %175 = icmp sgt i32 %163, 0
  %176 = zext nneg i32 %163 to i64
  %177 = lshr i64 %164, %176
  %178 = icmp eq i32 %163, -1
  %179 = zext i1 %178 to i64
  %180 = shl nuw nsw i64 %164, %179
  %181 = select i1 %175, i64 %177, i64 %180
  %182 = add nsw i32 %162, %150
  %183 = add nsw i32 %182, %163
  %184 = trunc i64 %181 to i32
  %185 = lshr i32 %184, 23
  %186 = and i32 %185, 1
  %187 = add nsw i32 %183, %186
  %188 = add nsw i32 %187, -1
  %189 = zext nneg i32 %165 to i64
  %190 = shl nuw nsw i64 1, %189
  %191 = add nsw i64 %190, -1
  %192 = and i64 %181, %190
  %193 = icmp eq i64 %192, 0
  %194 = select i1 %174, i1 %193, i1 false
  %195 = sext i1 %194 to i64
  %196 = add nsw i64 %181, %195
  %197 = and i64 %196, %191
  %198 = add nuw nsw i64 %197, %181
  %199 = icmp eq i32 %188, 0
  br i1 %199, label %200, label %204

200:                                              ; preds = %161
  %201 = trunc i64 %198 to i32
  %202 = lshr i32 %201, 23
  %203 = and i32 %202, 1
  br label %210

204:                                              ; preds = %161
  %205 = and i64 %198, 16777216
  %206 = icmp eq i64 %205, 0
  %207 = select i1 %206, i32 %188, i32 %187
  %208 = lshr exact i64 %205, 24
  %209 = lshr i64 %198, %208
  br label %210

210:                                              ; preds = %204, %200
  %211 = phi i32 [ %203, %200 ], [ %207, %204 ]
  %212 = phi i64 [ %198, %200 ], [ %209, %204 ]
  %213 = shl nsw i32 -1, %126
  %214 = xor i32 %213, -1
  %215 = icmp sgt i32 %211, %214
  br i1 %215, label %216, label %223

216:                                              ; preds = %210
  br i1 %6, label %217, label %221

217:                                              ; preds = %216
  %218 = shl nsw i32 -1, %127
  %219 = xor i32 %218, -1
  %220 = zext nneg i32 %219 to i64
  br label %234

221:                                              ; preds = %216
  %222 = trunc i32 %131 to i8
  br label %244

223:                                              ; preds = %210
  %224 = lshr i64 %212, %189
  %225 = icmp eq i32 %211, 0
  %226 = icmp eq i64 %224, 0
  %227 = select i1 %225, i1 %226, i1 false
  br i1 %227, label %232, label %228

228:                                              ; preds = %223
  %229 = shl nsw i32 -1, %127
  %230 = xor i32 %229, -1
  %231 = zext nneg i32 %230 to i64
  br label %234

232:                                              ; preds = %223
  %233 = trunc i32 %13 to i8
  br label %244

234:                                              ; preds = %228, %217
  %235 = phi i64 [ %231, %228 ], [ %220, %217 ]
  %236 = phi i64 [ %224, %228 ], [ %220, %217 ]
  %237 = phi i32 [ %211, %228 ], [ %214, %217 ]
  %238 = and i64 %236, %235
  %239 = shl nsw i32 %237, %127
  %240 = or i32 %239, %13
  %241 = zext i32 %240 to i64
  %242 = or i64 %238, %241
  %243 = trunc i64 %242 to i8
  br label %244

244:                                              ; preds = %22, %28, %30, %100, %105, %114, %134, %143, %145, %221, %232, %234
  %245 = phi i8 [ %23, %22 ], [ %29, %28 ], [ 0, %30 ], [ %123, %114 ], [ 0, %105 ], [ -128, %100 ], [ %138, %134 ], [ %144, %143 ], [ 0, %145 ], [ %233, %232 ], [ %243, %234 ], [ %222, %221 ]
  ret i8 %245
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef float @_Z20convert_fp8_to_floath26__hip_fp8_interpretation_t(i8 noundef zeroext %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = and i32 %1, -2
  %4 = icmp eq i32 %3, 2
  %5 = icmp eq i8 %0, 0
  br i1 %4, label %6, label %53

6:                                                ; preds = %2
  %7 = icmp eq i32 %1, 2
  %8 = select i1 %7, i32 3, i32 2
  br i1 %5, label %101, label %9

9:                                                ; preds = %6
  %10 = zext i8 %0 to i32
  %11 = shl nsw i32 -1, %8
  %12 = xor i32 %11, -1
  %13 = and i32 %12, %10
  %14 = zext nneg i32 %13 to i64
  %15 = icmp eq i8 %0, -128
  br i1 %15, label %101, label %16

16:                                               ; preds = %9
  %17 = and i32 %10, 127
  %18 = lshr i32 %17, %8
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %20, label %31

20:                                               ; preds = %16
  %21 = tail call i32 @llvm.ctlz.i32(i32 %13, i1 true), !range !13
  %22 = add nuw nsw i32 %21, %8
  %23 = add nsw i32 %22, -31
  %24 = zext nneg i32 %23 to i64
  %25 = shl nuw nsw i64 %14, %24
  %26 = sub nsw i32 32, %22
  %27 = zext nneg i32 %8 to i64
  %28 = shl nsw i64 -1, %27
  %29 = xor i64 %28, -1
  %30 = and i64 %25, %29
  br label %31

31:                                               ; preds = %20, %16
  %32 = phi i32 [ %26, %20 ], [ %18, %16 ]
  %33 = phi i64 [ %30, %20 ], [ %14, %16 ]
  %34 = select i1 %7, i32 7, i32 -1
  %35 = add nsw i32 %32, %34
  %36 = sub nuw nsw i32 10, %8
  %37 = zext nneg i32 %36 to i64
  %38 = shl nuw nsw i64 %33, %37
  %39 = icmp slt i32 %35, 1
  %40 = or i64 %38, 1024
  %41 = sub nsw i32 1, %35
  %42 = zext nneg i32 %41 to i64
  %43 = lshr i64 %40, %42
  %44 = shl nuw nsw i32 %35, 10
  %45 = select i1 %39, i64 %43, i64 %38
  %46 = shl nuw nsw i32 %10, 8
  %47 = and i32 %46, 32768
  %48 = select i1 %39, i32 0, i32 %44
  %49 = or i32 %48, %47
  %50 = zext nneg i32 %49 to i64
  %51 = or i64 %45, %50
  %52 = trunc i64 %51 to i16
  br label %101

53:                                               ; preds = %2
  %54 = icmp eq i32 %1, 0
  %55 = zext i8 %0 to i32
  br i1 %5, label %101, label %56

56:                                               ; preds = %53
  %57 = select i1 %54, i32 3, i32 2
  %58 = shl nsw i32 -1, %57
  %59 = xor i32 %58, -1
  %60 = and i32 %59, %55
  %61 = zext nneg i32 %60 to i64
  %62 = and i32 %55, 127
  %63 = lshr i32 %62, %57
  %64 = icmp eq i8 %0, -128
  br i1 %64, label %101, label %65

65:                                               ; preds = %56
  br i1 %54, label %66, label %68

66:                                               ; preds = %65
  %67 = icmp eq i32 %62, 127
  br i1 %67, label %101, label %80

68:                                               ; preds = %65
  %69 = and i32 %55, 124
  %70 = icmp eq i32 %69, 124
  br i1 %70, label %71, label %77

71:                                               ; preds = %68
  %72 = and i32 %55, 3
  %73 = icmp eq i32 %72, 0
  br i1 %73, label %74, label %101

74:                                               ; preds = %71
  %75 = icmp sgt i8 %0, -1
  %76 = select i1 %75, i16 31744, i16 -1024
  br label %101

77:                                               ; preds = %68
  %78 = zext i8 %0 to i16
  %79 = shl nuw i16 %78, 8
  br label %101

80:                                               ; preds = %66
  %81 = icmp eq i32 %63, 0
  br i1 %81, label %82, label %89

82:                                               ; preds = %80
  %83 = tail call i32 @llvm.ctlz.i32(i32 %60, i1 true), !range !13
  %84 = add nsw i32 %83, -28
  %85 = zext nneg i32 %84 to i64
  %86 = shl nuw nsw i64 %61, %85
  %87 = sub nsw i32 29, %83
  %88 = and i64 %86, 7
  br label %89

89:                                               ; preds = %82, %80
  %90 = phi i64 [ %88, %82 ], [ %61, %80 ]
  %91 = phi i32 [ %87, %82 ], [ %63, %80 ]
  %92 = shl nuw nsw i64 %90, 7
  %93 = shl nuw nsw i32 %55, 8
  %94 = and i32 %93, 32768
  %95 = shl nsw i32 %91, 10
  %96 = add nsw i32 %95, 8192
  %97 = or i32 %96, %94
  %98 = zext i32 %97 to i64
  %99 = or i64 %92, %98
  %100 = trunc i64 %99 to i16
  br label %101

101:                                              ; preds = %6, %9, %31, %53, %56, %66, %71, %74, %77, %89
  %102 = phi i16 [ 0, %6 ], [ %52, %31 ], [ 31745, %9 ], [ 0, %53 ], [ %76, %74 ], [ -32768, %56 ], [ 31745, %66 ], [ 31745, %71 ], [ %79, %77 ], [ %100, %89 ]
  %103 = bitcast i16 %102 to half
  %104 = fpext half %103 to float
  ret float %104
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #3

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  %6 = alloca %struct.dim3, align 8
  %7 = alloca %struct.dim3, align 8
  %8 = alloca i64, align 8
  %9 = alloca ptr, align 8
  %10 = alloca [5 x ptr], align 16
  %11 = alloca i8, align 1
  %12 = alloca i64, align 8
  %13 = alloca i64, align 8
  %14 = alloca i64, align 8
  %15 = alloca %struct.hipDeviceProp_tR0600, align 8
  %16 = alloca %"class.std::__cxx11::basic_string", align 8
  %17 = alloca %"class.std::__cxx11::basic_string", align 8
  %18 = alloca %"class.std::__cxx11::basic_string", align 8
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 1472, ptr nonnull %15) #16
  %21 = call i32 @hipGetDevicePropertiesR0600(ptr noundef nonnull %15, i32 noundef 0)
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %30, label %23

23:                                               ; preds = %0
  %24 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str)
  %25 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %24, ptr noundef nonnull @.str.1)
  %26 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %25, ptr noundef nonnull @.str.2)
  %27 = call ptr @hipGetErrorName(i32 noundef %21)
  %28 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef %27)
  %29 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr noundef nonnull align 8 dereferenceable(8) %28)
  call void @abort() #17
  unreachable

30:                                               ; preds = %0
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %16) #16
  %31 = getelementptr inbounds %struct.hipDeviceProp_tR0600, ptr %15, i64 0, i32 95
  %32 = getelementptr inbounds %"class.std::__cxx11::basic_string", ptr %16, i64 0, i32 2
  store ptr %32, ptr %16, align 8, !tbaa !14
  %33 = call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %31) #16
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %14) #16
  store i64 %33, ptr %14, align 8, !tbaa !11
  %34 = icmp ugt i64 %33, 15
  br i1 %34, label %35, label %39

35:                                               ; preds = %30
  %36 = invoke noundef ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(8) %14, i64 noundef 0)
          to label %37 unwind label %140

37:                                               ; preds = %35
  store ptr %36, ptr %16, align 8, !tbaa !16
  %38 = load i64, ptr %14, align 8, !tbaa !11
  store i64 %38, ptr %32, align 8, !tbaa !18
  br label %39

39:                                               ; preds = %37, %30
  %40 = phi ptr [ %36, %37 ], [ %32, %30 ]
  switch i64 %33, label %43 [
    i64 1, label %41
    i64 0, label %44
  ]

41:                                               ; preds = %39
  %42 = load i8, ptr %31, align 8, !tbaa !18
  store i8 %42, ptr %40, align 1, !tbaa !18
  br label %44

43:                                               ; preds = %39
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %40, ptr nonnull align 8 %31, i64 %33, i1 false)
  br label %44

44:                                               ; preds = %43, %41, %39
  %45 = load i64, ptr %14, align 8, !tbaa !11
  %46 = getelementptr inbounds %"class.std::__cxx11::basic_string", ptr %16, i64 0, i32 1
  store i64 %45, ptr %46, align 8, !tbaa !19
  %47 = load ptr, ptr %16, align 8, !tbaa !16
  %48 = getelementptr inbounds i8, ptr %47, i64 %45
  store i8 0, ptr %48, align 1, !tbaa !18
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %14) #16
  %49 = call noundef i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4findEPKcmm(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull @.str.3, i64 noundef 0, i64 noundef 5) #16
  %50 = icmp eq i64 %49, -1
  br i1 %50, label %51, label %78

51:                                               ; preds = %44
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %17) #16
  %52 = getelementptr inbounds %"class.std::__cxx11::basic_string", ptr %17, i64 0, i32 2
  store ptr %52, ptr %17, align 8, !tbaa !14
  %53 = call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %31) #16
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %13) #16
  store i64 %53, ptr %13, align 8, !tbaa !11
  %54 = icmp ugt i64 %53, 15
  br i1 %54, label %55, label %59

55:                                               ; preds = %51
  %56 = invoke noundef ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(8) %13, i64 noundef 0)
          to label %57 unwind label %142

57:                                               ; preds = %55
  store ptr %56, ptr %17, align 8, !tbaa !16
  %58 = load i64, ptr %13, align 8, !tbaa !11
  store i64 %58, ptr %52, align 8, !tbaa !18
  br label %59

59:                                               ; preds = %57, %51
  %60 = phi ptr [ %56, %57 ], [ %52, %51 ]
  switch i64 %53, label %63 [
    i64 1, label %61
    i64 0, label %64
  ]

61:                                               ; preds = %59
  %62 = load i8, ptr %31, align 8, !tbaa !18
  store i8 %62, ptr %60, align 1, !tbaa !18
  br label %64

63:                                               ; preds = %59
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %60, ptr nonnull align 8 %31, i64 %53, i1 false)
  br label %64

64:                                               ; preds = %59, %61, %63
  %65 = load i64, ptr %13, align 8, !tbaa !11
  %66 = getelementptr inbounds %"class.std::__cxx11::basic_string", ptr %17, i64 0, i32 1
  store i64 %65, ptr %66, align 8, !tbaa !19
  %67 = load ptr, ptr %17, align 8, !tbaa !16
  %68 = getelementptr inbounds i8, ptr %67, i64 %65
  store i8 0, ptr %68, align 1, !tbaa !18
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %13) #16
  %69 = call noundef i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4findEPKcmm(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull @.str.4, i64 noundef 0, i64 noundef 6) #16
  %70 = icmp ne i64 %69, -1
  %71 = load ptr, ptr %17, align 8, !tbaa !16
  %72 = icmp eq ptr %71, %52
  br i1 %72, label %73, label %76

73:                                               ; preds = %64
  %74 = load i64, ptr %66, align 8, !tbaa !19
  %75 = icmp ult i64 %74, 16
  call void @llvm.assume(i1 %75)
  br label %77

76:                                               ; preds = %64
  call void @_ZdlPv(ptr noundef %71) #18
  br label %77

77:                                               ; preds = %73, %76
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %17) #16
  br label %78

78:                                               ; preds = %44, %77
  %79 = phi i1 [ %70, %77 ], [ true, %44 ]
  %80 = load ptr, ptr %16, align 8, !tbaa !16
  %81 = icmp eq ptr %80, %32
  br i1 %81, label %82, label %85

82:                                               ; preds = %78
  %83 = load i64, ptr %46, align 8, !tbaa !19
  %84 = icmp ult i64 %83, 16
  call void @llvm.assume(i1 %84)
  br label %86

85:                                               ; preds = %78
  call void @_ZdlPv(ptr noundef %80) #18
  br label %86

86:                                               ; preds = %82, %85
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %16) #16
  br i1 %79, label %152, label %87

87:                                               ; preds = %86
  %88 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.5, i64 noundef 37)
  %89 = call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %31) #16
  %90 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull %31, i64 noundef %89)
  %91 = load ptr, ptr @_ZSt4cerr, align 8, !tbaa !20
  %92 = getelementptr i8, ptr %91, i64 -24
  %93 = load i64, ptr %92, align 8
  %94 = getelementptr inbounds i8, ptr @_ZSt4cerr, i64 %93
  %95 = getelementptr inbounds %"class.std::basic_ios", ptr %94, i64 0, i32 5
  %96 = load ptr, ptr %95, align 8, !tbaa !22
  %97 = icmp eq ptr %96, null
  br i1 %97, label %98, label %99

98:                                               ; preds = %87
  call void @_ZSt16__throw_bad_castv() #19
  unreachable

99:                                               ; preds = %87
  %100 = getelementptr inbounds %"class.std::ctype", ptr %96, i64 0, i32 8
  %101 = load i8, ptr %100, align 8, !tbaa !31
  %102 = icmp eq i8 %101, 0
  br i1 %102, label %106, label %103

103:                                              ; preds = %99
  %104 = getelementptr inbounds %"class.std::ctype", ptr %96, i64 0, i32 9, i64 10
  %105 = load i8, ptr %104, align 1, !tbaa !18
  br label %111

106:                                              ; preds = %99
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %96)
  %107 = load ptr, ptr %96, align 8, !tbaa !20
  %108 = getelementptr inbounds ptr, ptr %107, i64 6
  %109 = load ptr, ptr %108, align 8
  %110 = call noundef signext i8 %109(ptr noundef nonnull align 8 dereferenceable(570) %96, i8 noundef signext 10)
  br label %111

111:                                              ; preds = %103, %106
  %112 = phi i8 [ %105, %103 ], [ %110, %106 ]
  %113 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i8 noundef signext %112)
  %114 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %113)
  %115 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.6, i64 noundef 73)
  %116 = load ptr, ptr @_ZSt4cerr, align 8, !tbaa !20
  %117 = getelementptr i8, ptr %116, i64 -24
  %118 = load i64, ptr %117, align 8
  %119 = getelementptr inbounds i8, ptr @_ZSt4cerr, i64 %118
  %120 = getelementptr inbounds %"class.std::basic_ios", ptr %119, i64 0, i32 5
  %121 = load ptr, ptr %120, align 8, !tbaa !22
  %122 = icmp eq ptr %121, null
  br i1 %122, label %123, label %124

123:                                              ; preds = %111
  call void @_ZSt16__throw_bad_castv() #19
  unreachable

124:                                              ; preds = %111
  %125 = getelementptr inbounds %"class.std::ctype", ptr %121, i64 0, i32 8
  %126 = load i8, ptr %125, align 8, !tbaa !31
  %127 = icmp eq i8 %126, 0
  br i1 %127, label %131, label %128

128:                                              ; preds = %124
  %129 = getelementptr inbounds %"class.std::ctype", ptr %121, i64 0, i32 9, i64 10
  %130 = load i8, ptr %129, align 1, !tbaa !18
  br label %136

131:                                              ; preds = %124
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %121)
  %132 = load ptr, ptr %121, align 8, !tbaa !20
  %133 = getelementptr inbounds ptr, ptr %132, i64 6
  %134 = load ptr, ptr %133, align 8
  %135 = call noundef signext i8 %134(ptr noundef nonnull align 8 dereferenceable(570) %121, i8 noundef signext 10)
  br label %136

136:                                              ; preds = %128, %131
  %137 = phi i8 [ %130, %128 ], [ %135, %131 ]
  %138 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i8 noundef signext %137)
  %139 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %138)
  br label %714

140:                                              ; preds = %35
  %141 = landingpad { ptr, i32 }
          cleanup
  br label %150

142:                                              ; preds = %55
  %143 = landingpad { ptr, i32 }
          cleanup
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %17) #16
  %144 = load ptr, ptr %16, align 8, !tbaa !16
  %145 = icmp eq ptr %144, %32
  br i1 %145, label %146, label %149

146:                                              ; preds = %142
  %147 = load i64, ptr %46, align 8, !tbaa !19
  %148 = icmp ult i64 %147, 16
  call void @llvm.assume(i1 %148)
  br label %150

149:                                              ; preds = %142
  call void @_ZdlPv(ptr noundef %144) #18
  br label %150

150:                                              ; preds = %149, %146, %140
  %151 = phi { ptr, i32 } [ %141, %140 ], [ %143, %146 ], [ %143, %149 ]
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %16) #16
  br label %728

152:                                              ; preds = %86
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %18) #16
  %153 = getelementptr inbounds %"class.std::__cxx11::basic_string", ptr %18, i64 0, i32 2
  store ptr %153, ptr %18, align 8, !tbaa !14
  %154 = call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %31) #16
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %12) #16
  store i64 %154, ptr %12, align 8, !tbaa !11
  %155 = icmp ugt i64 %154, 15
  br i1 %155, label %156, label %159

156:                                              ; preds = %152
  %157 = call noundef ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(8) %12, i64 noundef 0)
  store ptr %157, ptr %18, align 8, !tbaa !16
  %158 = load i64, ptr %12, align 8, !tbaa !11
  store i64 %158, ptr %153, align 8, !tbaa !18
  br label %159

159:                                              ; preds = %156, %152
  %160 = phi ptr [ %157, %156 ], [ %153, %152 ]
  switch i64 %154, label %163 [
    i64 1, label %161
    i64 0, label %164
  ]

161:                                              ; preds = %159
  %162 = load i8, ptr %31, align 8, !tbaa !18
  store i8 %162, ptr %160, align 1, !tbaa !18
  br label %164

163:                                              ; preds = %159
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %160, ptr nonnull align 8 %31, i64 %154, i1 false)
  br label %164

164:                                              ; preds = %163, %161, %159
  %165 = load i64, ptr %12, align 8, !tbaa !11
  %166 = getelementptr inbounds %"class.std::__cxx11::basic_string", ptr %18, i64 0, i32 1
  store i64 %165, ptr %166, align 8, !tbaa !19
  %167 = load ptr, ptr %18, align 8, !tbaa !16
  %168 = getelementptr inbounds i8, ptr %167, i64 %165
  store i8 0, ptr %168, align 1, !tbaa !18
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %12) #16
  %169 = call noundef i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4findEPKcmm(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull @.str.3, i64 noundef 0, i64 noundef 5) #16
  %170 = load ptr, ptr %18, align 8, !tbaa !16
  %171 = icmp eq ptr %170, %153
  br i1 %171, label %172, label %175

172:                                              ; preds = %164
  %173 = load i64, ptr %166, align 8, !tbaa !19
  %174 = icmp ult i64 %173, 16
  call void @llvm.assume(i1 %174)
  br label %176

175:                                              ; preds = %164
  call void @_ZdlPv(ptr noundef %170) #18
  br label %176

176:                                              ; preds = %175, %172
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %18) #16
  %177 = call noalias noundef nonnull dereferenceable(128) ptr @_Znwm(i64 noundef 128) #20
  %178 = getelementptr inbounds float, ptr %177, i64 1
  store <2 x float> <float 1.000000e+00, float 3.250000e+00>, ptr %177, align 4, !tbaa !34
  %179 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.7, i64 noundef 15)
          to label %180 unwind label %724

180:                                              ; preds = %176
  %181 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, double noundef 1.000000e+00)
          to label %182 unwind label %724

182:                                              ; preds = %180
  %183 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %181, ptr noundef nonnull @.str.8, i64 noundef 1)
          to label %184 unwind label %724

184:                                              ; preds = %182
  %185 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.9, i64 noundef 35)
          to label %186 unwind label %724

186:                                              ; preds = %184
  %187 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !20
  %188 = getelementptr i8, ptr %187, i64 -24
  %189 = load i64, ptr %188, align 8
  %190 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %189
  %191 = getelementptr inbounds %"class.std::basic_ios", ptr %190, i64 0, i32 5
  %192 = load ptr, ptr %191, align 8, !tbaa !22
  %193 = icmp eq ptr %192, null
  br i1 %193, label %194, label %196

194:                                              ; preds = %186
  invoke void @_ZSt16__throw_bad_castv() #19
          to label %195 unwind label %724

195:                                              ; preds = %194
  unreachable

196:                                              ; preds = %186
  %197 = getelementptr inbounds %"class.std::ctype", ptr %192, i64 0, i32 8
  %198 = load i8, ptr %197, align 8, !tbaa !31
  %199 = icmp eq i8 %198, 0
  br i1 %199, label %203, label %200

200:                                              ; preds = %196
  %201 = getelementptr inbounds %"class.std::ctype", ptr %192, i64 0, i32 9, i64 10
  %202 = load i8, ptr %201, align 1, !tbaa !18
  br label %209

203:                                              ; preds = %196
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %192)
          to label %204 unwind label %724

204:                                              ; preds = %203
  %205 = load ptr, ptr %192, align 8, !tbaa !20
  %206 = getelementptr inbounds ptr, ptr %205, i64 6
  %207 = load ptr, ptr %206, align 8
  %208 = invoke noundef signext i8 %207(ptr noundef nonnull align 8 dereferenceable(570) %192, i8 noundef signext 10)
          to label %209 unwind label %724

209:                                              ; preds = %204, %200
  %210 = phi i8 [ %202, %200 ], [ %208, %204 ]
  %211 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef signext %210)
          to label %212 unwind label %724

212:                                              ; preds = %209
  %213 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %211)
          to label %214 unwind label %724

214:                                              ; preds = %212
  %215 = invoke noalias noundef nonnull dereferenceable(128) ptr @_Znwm(i64 noundef 128) #20
          to label %216 unwind label %220

216:                                              ; preds = %214
  %217 = load float, ptr %177, align 4, !tbaa !34
  %218 = call noundef zeroext i8 @_Z20convert_float_to_fp8f26__hip_fp8_interpretation_t18__hip_saturation_t(float noundef %217, i32 noundef 0, i32 noundef 1)
  %219 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.10, i64 noundef 5)
          to label %222 unwind label %327

220:                                              ; preds = %214
  %221 = landingpad { ptr, i32 }
          cleanup
  br label %726

222:                                              ; preds = %216
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %11)
  store i8 %218, ptr %11, align 1, !tbaa !18
  %223 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !20
  %224 = getelementptr i8, ptr %223, i64 -24
  %225 = load i64, ptr %224, align 8
  %226 = getelementptr i8, ptr getelementptr inbounds (%"class.std::basic_ostream", ptr @_ZSt4cout, i64 0, i32 1, i32 0, i32 1), i64 %225
  %227 = load i64, ptr %226, align 8, !tbaa !36
  %228 = icmp eq i64 %227, 0
  br i1 %228, label %231, label %229

229:                                              ; preds = %222
  %230 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %11, i64 noundef 1)
          to label %233 unwind label %327

231:                                              ; preds = %222
  %232 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef signext %218)
          to label %233 unwind label %327

233:                                              ; preds = %229, %231
  %234 = phi ptr [ %230, %229 ], [ @_ZSt4cout, %231 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %11)
  %235 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %234, ptr noundef nonnull @.str.8, i64 noundef 1)
          to label %236 unwind label %327

236:                                              ; preds = %233
  %237 = icmp eq i8 %218, 0
  %238 = zext i8 %218 to i32
  br i1 %237, label %268, label %239

239:                                              ; preds = %236
  %240 = and i32 %238, 7
  %241 = zext nneg i32 %240 to i64
  %242 = and i32 %238, 127
  %243 = lshr i32 %242, 3
  %244 = icmp eq i8 %218, -128
  br i1 %244, label %268, label %245

245:                                              ; preds = %239
  %246 = icmp eq i32 %242, 127
  br i1 %246, label %268, label %247

247:                                              ; preds = %245
  %248 = icmp ult i32 %242, 8
  br i1 %248, label %249, label %256

249:                                              ; preds = %247
  %250 = call i32 @llvm.ctlz.i32(i32 %240, i1 true), !range !13
  %251 = add nsw i32 %250, -28
  %252 = zext nneg i32 %251 to i64
  %253 = shl nuw nsw i64 %241, %252
  %254 = sub nsw i32 29, %250
  %255 = and i64 %253, 7
  br label %256

256:                                              ; preds = %249, %247
  %257 = phi i64 [ %255, %249 ], [ %241, %247 ]
  %258 = phi i32 [ %254, %249 ], [ %243, %247 ]
  %259 = shl nuw nsw i64 %257, 7
  %260 = shl nuw nsw i32 %238, 8
  %261 = and i32 %260, 32768
  %262 = shl nsw i32 %258, 10
  %263 = add nsw i32 %262, 8192
  %264 = or i32 %263, %261
  %265 = zext i32 %264 to i64
  %266 = or i64 %259, %265
  %267 = trunc i64 %266 to i16
  br label %268

268:                                              ; preds = %256, %245, %239, %236
  %269 = phi i16 [ 0, %236 ], [ -32768, %239 ], [ 31745, %245 ], [ %267, %256 ]
  %270 = bitcast i16 %269 to half
  %271 = fpext half %270 to float
  store float %271, ptr %215, align 4, !tbaa !34
  %272 = getelementptr inbounds float, ptr %215, i64 1
  %273 = load float, ptr %178, align 4, !tbaa !34
  %274 = call noundef zeroext i8 @_Z20convert_float_to_fp8f26__hip_fp8_interpretation_t18__hip_saturation_t(float noundef %273, i32 noundef 0, i32 noundef 1)
  %275 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.10, i64 noundef 5)
          to label %276 unwind label %327

276:                                              ; preds = %268
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %11)
  store i8 %274, ptr %11, align 1, !tbaa !18
  %277 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !20
  %278 = getelementptr i8, ptr %277, i64 -24
  %279 = load i64, ptr %278, align 8
  %280 = getelementptr i8, ptr getelementptr inbounds (%"class.std::basic_ostream", ptr @_ZSt4cout, i64 0, i32 1, i32 0, i32 1), i64 %279
  %281 = load i64, ptr %280, align 8, !tbaa !36
  %282 = icmp eq i64 %281, 0
  br i1 %282, label %285, label %283

283:                                              ; preds = %276
  %284 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %11, i64 noundef 1)
          to label %287 unwind label %327

285:                                              ; preds = %276
  %286 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef signext %274)
          to label %287 unwind label %327

287:                                              ; preds = %285, %283
  %288 = phi ptr [ %284, %283 ], [ @_ZSt4cout, %285 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %11)
  %289 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %288, ptr noundef nonnull @.str.8, i64 noundef 1)
          to label %290 unwind label %327

290:                                              ; preds = %287
  %291 = icmp eq i8 %274, 0
  %292 = zext i8 %274 to i32
  br i1 %291, label %322, label %293

293:                                              ; preds = %290
  %294 = and i32 %292, 7
  %295 = zext nneg i32 %294 to i64
  %296 = and i32 %292, 127
  %297 = lshr i32 %296, 3
  %298 = icmp eq i8 %274, -128
  br i1 %298, label %322, label %299

299:                                              ; preds = %293
  %300 = icmp eq i32 %296, 127
  br i1 %300, label %322, label %301

301:                                              ; preds = %299
  %302 = icmp ult i32 %296, 8
  br i1 %302, label %303, label %310

303:                                              ; preds = %301
  %304 = call i32 @llvm.ctlz.i32(i32 %294, i1 true), !range !13
  %305 = add nsw i32 %304, -28
  %306 = zext nneg i32 %305 to i64
  %307 = shl nuw nsw i64 %295, %306
  %308 = sub nsw i32 29, %304
  %309 = and i64 %307, 7
  br label %310

310:                                              ; preds = %303, %301
  %311 = phi i64 [ %309, %303 ], [ %295, %301 ]
  %312 = phi i32 [ %308, %303 ], [ %297, %301 ]
  %313 = shl nuw nsw i64 %311, 7
  %314 = shl nuw nsw i32 %292, 8
  %315 = and i32 %314, 32768
  %316 = shl nsw i32 %312, 10
  %317 = add nsw i32 %316, 8192
  %318 = or i32 %317, %315
  %319 = zext i32 %318 to i64
  %320 = or i64 %313, %319
  %321 = trunc i64 %320 to i16
  br label %322

322:                                              ; preds = %310, %299, %293, %290
  %323 = phi i16 [ 0, %290 ], [ -32768, %293 ], [ 31745, %299 ], [ %321, %310 ]
  %324 = bitcast i16 %323 to half
  %325 = fpext half %324 to float
  store float %325, ptr %272, align 4, !tbaa !34
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %19) #16
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %20) #16
  %326 = invoke noundef i32 @hipMalloc(ptr noundef nonnull %19, i64 noundef 128)
          to label %329 unwind label %344

327:                                              ; preds = %287, %285, %283, %268, %233, %231, %229, %216
  %328 = landingpad { ptr, i32 }
          cleanup
  br label %722

329:                                              ; preds = %322
  %330 = icmp eq i32 %326, 0
  br i1 %330, label %346, label %331

331:                                              ; preds = %329
  %332 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str, i64 noundef 20)
          to label %333 unwind label %344

333:                                              ; preds = %331
  %334 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.11, i64 noundef 38)
          to label %335 unwind label %344

335:                                              ; preds = %333
  %336 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.2, i64 noundef 13)
          to label %337 unwind label %344

337:                                              ; preds = %335
  %338 = invoke ptr @hipGetErrorName(i32 noundef %326)
          to label %339 unwind label %344

339:                                              ; preds = %337
  %340 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef %338)
          to label %341 unwind label %344

341:                                              ; preds = %339
  %342 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr noundef nonnull align 8 dereferenceable(8) %340)
          to label %343 unwind label %344

343:                                              ; preds = %341
  call void @abort() #17
  unreachable

344:                                              ; preds = %341, %335, %333, %331, %322, %339, %337
  %345 = landingpad { ptr, i32 }
          cleanup
  br label %720

346:                                              ; preds = %329
  %347 = invoke noundef i32 @hipMalloc(ptr noundef nonnull %20, i64 noundef 128)
          to label %348 unwind label %363

348:                                              ; preds = %346
  %349 = icmp eq i32 %347, 0
  br i1 %349, label %365, label %350

350:                                              ; preds = %348
  %351 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str, i64 noundef 20)
          to label %352 unwind label %363

352:                                              ; preds = %350
  %353 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.12, i64 noundef 39)
          to label %354 unwind label %363

354:                                              ; preds = %352
  %355 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.2, i64 noundef 13)
          to label %356 unwind label %363

356:                                              ; preds = %354
  %357 = invoke ptr @hipGetErrorName(i32 noundef %347)
          to label %358 unwind label %363

358:                                              ; preds = %356
  %359 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef %357)
          to label %360 unwind label %363

360:                                              ; preds = %358
  %361 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr noundef nonnull align 8 dereferenceable(8) %359)
          to label %362 unwind label %363

362:                                              ; preds = %360
  call void @abort() #17
  unreachable

363:                                              ; preds = %360, %354, %352, %350, %346, %358, %356
  %364 = landingpad { ptr, i32 }
          cleanup
  br label %720

365:                                              ; preds = %348
  %366 = load ptr, ptr %19, align 8, !tbaa !3
  %367 = invoke i32 @hipMemcpy(ptr noundef %366, ptr noundef nonnull %177, i64 noundef 8, i32 noundef 1)
          to label %368 unwind label %383

368:                                              ; preds = %365
  %369 = icmp eq i32 %367, 0
  br i1 %369, label %385, label %370

370:                                              ; preds = %368
  %371 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str, i64 noundef 20)
          to label %372 unwind label %383

372:                                              ; preds = %370
  %373 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.13, i64 noundef 76)
          to label %374 unwind label %383

374:                                              ; preds = %372
  %375 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.2, i64 noundef 13)
          to label %376 unwind label %383

376:                                              ; preds = %374
  %377 = invoke ptr @hipGetErrorName(i32 noundef %367)
          to label %378 unwind label %383

378:                                              ; preds = %376
  %379 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef %377)
          to label %380 unwind label %383

380:                                              ; preds = %378
  %381 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr noundef nonnull align 8 dereferenceable(8) %379)
          to label %382 unwind label %383

382:                                              ; preds = %380
  call void @abort() #17
  unreachable

383:                                              ; preds = %380, %374, %372, %370, %378, %376, %365
  %384 = landingpad { ptr, i32 }
          cleanup
  br label %720

385:                                              ; preds = %368
  %386 = invoke i32 @__hipPushCallConfiguration(i64 4294967297, i32 1, i64 4294967328, i32 1, i64 noundef 0, ptr noundef null)
          to label %387 unwind label %408

387:                                              ; preds = %385
  %388 = icmp eq i32 %386, 0
  br i1 %388, label %389, label %410

389:                                              ; preds = %387
  %390 = load ptr, ptr %19, align 8, !tbaa !3
  %391 = load ptr, ptr %20, align 8, !tbaa !3
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %1)
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %2)
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %3)
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %4)
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %5)
  call void @llvm.lifetime.start.p0(i64 12, ptr nonnull %6)
  call void @llvm.lifetime.start.p0(i64 12, ptr nonnull %7)
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %8)
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %9)
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %10)
  store ptr %390, ptr %1, align 8, !tbaa !3
  store i32 0, ptr %2, align 4, !tbaa !7
  store i32 1, ptr %3, align 4, !tbaa !9
  store ptr %391, ptr %4, align 8, !tbaa !3
  store i64 32, ptr %5, align 8, !tbaa !11
  store ptr %1, ptr %10, align 16
  %392 = getelementptr inbounds ptr, ptr %10, i64 1
  store ptr %2, ptr %392, align 8
  %393 = getelementptr inbounds ptr, ptr %10, i64 2
  store ptr %3, ptr %393, align 16
  %394 = getelementptr inbounds ptr, ptr %10, i64 3
  store ptr %4, ptr %394, align 8
  %395 = getelementptr inbounds ptr, ptr %10, i64 4
  store ptr %5, ptr %395, align 16
  %396 = invoke i32 @__hipPopCallConfiguration(ptr nonnull %6, ptr nonnull %7, ptr nonnull %8, ptr nonnull %9)
          to label %397 unwind label %408

397:                                              ; preds = %389
  %398 = load i64, ptr %8, align 8
  %399 = load ptr, ptr %9, align 8
  %400 = load i64, ptr %6, align 8
  %401 = getelementptr inbounds i8, ptr %6, i64 8
  %402 = load i32, ptr %401, align 8
  %403 = load i64, ptr %7, align 8
  %404 = getelementptr inbounds i8, ptr %7, i64 8
  %405 = load i32, ptr %404, align 8
  %406 = invoke noundef i32 @hipLaunchKernel(ptr noundef nonnull @_Z21float_to_fp8_to_floatPf26__hip_fp8_interpretation_t18__hip_saturation_tS_m, i64 %400, i32 %402, i64 %403, i32 %405, ptr noundef nonnull %10, i64 noundef %398, ptr noundef %399)
          to label %407 unwind label %408

407:                                              ; preds = %397
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %1)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %2)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %3)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %4)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %5)
  call void @llvm.lifetime.end.p0(i64 12, ptr nonnull %6)
  call void @llvm.lifetime.end.p0(i64 12, ptr nonnull %7)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %8)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %9)
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %10)
  br label %410

408:                                              ; preds = %397, %389, %385
  %409 = landingpad { ptr, i32 }
          cleanup
  br label %720

410:                                              ; preds = %407, %387
  %411 = invoke noalias noundef nonnull dereferenceable(128) ptr @_Znwm(i64 noundef 128) #20
          to label %412 unwind label %430

412:                                              ; preds = %410
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(128) %411, i8 0, i64 128, i1 false), !tbaa !34
  %413 = load ptr, ptr %20, align 8, !tbaa !3
  %414 = invoke i32 @hipMemcpy(ptr noundef nonnull %411, ptr noundef %413, i64 noundef 128, i32 noundef 2)
          to label %415 unwind label %432

415:                                              ; preds = %412
  %416 = icmp eq i32 %414, 0
  br i1 %416, label %434, label %417

417:                                              ; preds = %415
  %418 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str, i64 noundef 20)
          to label %419 unwind label %432

419:                                              ; preds = %417
  %420 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14, i64 noundef 87)
          to label %421 unwind label %432

421:                                              ; preds = %419
  %422 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.2, i64 noundef 13)
          to label %423 unwind label %432

423:                                              ; preds = %421
  %424 = invoke ptr @hipGetErrorName(i32 noundef %414)
          to label %425 unwind label %432

425:                                              ; preds = %423
  %426 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef %424)
          to label %427 unwind label %432

427:                                              ; preds = %425
  %428 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr noundef nonnull align 8 dereferenceable(8) %426)
          to label %429 unwind label %432

429:                                              ; preds = %427
  call void @abort() #17
  unreachable

430:                                              ; preds = %410
  %431 = landingpad { ptr, i32 }
          cleanup
  br label %720

432:                                              ; preds = %427, %421, %419, %417, %425, %423, %412
  %433 = landingpad { ptr, i32 }
          cleanup
  br label %718

434:                                              ; preds = %415
  %435 = load ptr, ptr %19, align 8, !tbaa !3
  %436 = invoke i32 @hipFree(ptr noundef %435)
          to label %437 unwind label %452

437:                                              ; preds = %434
  %438 = icmp eq i32 %436, 0
  br i1 %438, label %454, label %439

439:                                              ; preds = %437
  %440 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str, i64 noundef 20)
          to label %441 unwind label %452

441:                                              ; preds = %439
  %442 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.15, i64 noundef 13)
          to label %443 unwind label %452

443:                                              ; preds = %441
  %444 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.2, i64 noundef 13)
          to label %445 unwind label %452

445:                                              ; preds = %443
  %446 = invoke ptr @hipGetErrorName(i32 noundef %436)
          to label %447 unwind label %452

447:                                              ; preds = %445
  %448 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef %446)
          to label %449 unwind label %452

449:                                              ; preds = %447
  %450 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr noundef nonnull align 8 dereferenceable(8) %448)
          to label %451 unwind label %452

451:                                              ; preds = %449
  call void @abort() #17
  unreachable

452:                                              ; preds = %449, %443, %441, %439, %447, %445, %434
  %453 = landingpad { ptr, i32 }
          cleanup
  br label %718

454:                                              ; preds = %437
  %455 = load ptr, ptr %20, align 8, !tbaa !3
  %456 = invoke i32 @hipFree(ptr noundef %455)
          to label %457 unwind label %476

457:                                              ; preds = %454
  %458 = icmp eq i32 %456, 0
  br i1 %458, label %459, label %463

459:                                              ; preds = %457
  %460 = load float, ptr %215, align 4, !tbaa !34
  %461 = load float, ptr %411, align 4, !tbaa !34
  %462 = fcmp contract une float %460, %461
  br i1 %462, label %478, label %498

463:                                              ; preds = %457
  %464 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str, i64 noundef 20)
          to label %465 unwind label %476

465:                                              ; preds = %463
  %466 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.16, i64 noundef 14)
          to label %467 unwind label %476

467:                                              ; preds = %465
  %468 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.2, i64 noundef 13)
          to label %469 unwind label %476

469:                                              ; preds = %467
  %470 = invoke ptr @hipGetErrorName(i32 noundef %456)
          to label %471 unwind label %476

471:                                              ; preds = %469
  %472 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef %470)
          to label %473 unwind label %476

473:                                              ; preds = %471
  %474 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr noundef nonnull align 8 dereferenceable(8) %472)
          to label %475 unwind label %476

475:                                              ; preds = %473
  call void @abort() #17
  unreachable

476:                                              ; preds = %473, %467, %465, %463, %471, %469, %454
  %477 = landingpad { ptr, i32 }
          cleanup
  br label %718

478:                                              ; preds = %677, %671, %665, %659, %653, %647, %641, %635, %629, %623, %617, %611, %605, %599, %593, %587, %581, %575, %569, %563, %557, %551, %545, %539, %533, %527, %521, %515, %509, %503, %498, %459
  %479 = phi i64 [ 0, %459 ], [ 1, %498 ], [ 2, %503 ], [ 3, %509 ], [ 4, %515 ], [ 5, %521 ], [ 6, %527 ], [ 7, %533 ], [ 8, %539 ], [ 9, %545 ], [ 10, %551 ], [ 11, %557 ], [ 12, %563 ], [ 13, %569 ], [ 14, %575 ], [ 15, %581 ], [ 16, %587 ], [ 17, %593 ], [ 18, %599 ], [ 19, %605 ], [ 20, %611 ], [ 21, %617 ], [ 22, %623 ], [ 23, %629 ], [ 24, %635 ], [ 25, %641 ], [ 26, %647 ], [ 27, %653 ], [ 28, %659 ], [ 29, %665 ], [ 30, %671 ], [ 31, %677 ]
  %480 = getelementptr inbounds float, ptr %411, i64 %479
  %481 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.17, i64 noundef 23)
          to label %482 unwind label %496

482:                                              ; preds = %478
  %483 = getelementptr inbounds float, ptr %215, i64 %479
  %484 = load float, ptr %483, align 4, !tbaa !34
  %485 = fpext float %484 to double
  %486 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, double noundef %485)
          to label %487 unwind label %496

487:                                              ; preds = %482
  %488 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %486, ptr noundef nonnull @.str.18, i64 noundef 26)
          to label %489 unwind label %496

489:                                              ; preds = %487
  %490 = load float, ptr %480, align 4, !tbaa !34
  %491 = fpext float %490 to double
  %492 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %486, double noundef %491)
          to label %493 unwind label %496

493:                                              ; preds = %489
  %494 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr noundef nonnull align 8 dereferenceable(8) %492)
          to label %495 unwind label %496

495:                                              ; preds = %493
  call void @abort() #17
  unreachable

496:                                              ; preds = %493, %489, %487, %482, %478
  %497 = landingpad { ptr, i32 }
          cleanup
  br label %718

498:                                              ; preds = %459
  %499 = load float, ptr %272, align 4, !tbaa !34
  %500 = getelementptr inbounds float, ptr %411, i64 1
  %501 = load float, ptr %500, align 4, !tbaa !34
  %502 = fcmp contract une float %499, %501
  br i1 %502, label %478, label %503

503:                                              ; preds = %498
  %504 = getelementptr inbounds float, ptr %215, i64 2
  %505 = load float, ptr %504, align 4, !tbaa !34
  %506 = getelementptr inbounds float, ptr %411, i64 2
  %507 = load float, ptr %506, align 4, !tbaa !34
  %508 = fcmp contract une float %505, %507
  br i1 %508, label %478, label %509

509:                                              ; preds = %503
  %510 = getelementptr inbounds float, ptr %215, i64 3
  %511 = load float, ptr %510, align 4, !tbaa !34
  %512 = getelementptr inbounds float, ptr %411, i64 3
  %513 = load float, ptr %512, align 4, !tbaa !34
  %514 = fcmp contract une float %511, %513
  br i1 %514, label %478, label %515

515:                                              ; preds = %509
  %516 = getelementptr inbounds float, ptr %215, i64 4
  %517 = load float, ptr %516, align 4, !tbaa !34
  %518 = getelementptr inbounds float, ptr %411, i64 4
  %519 = load float, ptr %518, align 4, !tbaa !34
  %520 = fcmp contract une float %517, %519
  br i1 %520, label %478, label %521

521:                                              ; preds = %515
  %522 = getelementptr inbounds float, ptr %215, i64 5
  %523 = load float, ptr %522, align 4, !tbaa !34
  %524 = getelementptr inbounds float, ptr %411, i64 5
  %525 = load float, ptr %524, align 4, !tbaa !34
  %526 = fcmp contract une float %523, %525
  br i1 %526, label %478, label %527

527:                                              ; preds = %521
  %528 = getelementptr inbounds float, ptr %215, i64 6
  %529 = load float, ptr %528, align 4, !tbaa !34
  %530 = getelementptr inbounds float, ptr %411, i64 6
  %531 = load float, ptr %530, align 4, !tbaa !34
  %532 = fcmp contract une float %529, %531
  br i1 %532, label %478, label %533

533:                                              ; preds = %527
  %534 = getelementptr inbounds float, ptr %215, i64 7
  %535 = load float, ptr %534, align 4, !tbaa !34
  %536 = getelementptr inbounds float, ptr %411, i64 7
  %537 = load float, ptr %536, align 4, !tbaa !34
  %538 = fcmp contract une float %535, %537
  br i1 %538, label %478, label %539

539:                                              ; preds = %533
  %540 = getelementptr inbounds float, ptr %215, i64 8
  %541 = load float, ptr %540, align 4, !tbaa !34
  %542 = getelementptr inbounds float, ptr %411, i64 8
  %543 = load float, ptr %542, align 4, !tbaa !34
  %544 = fcmp contract une float %541, %543
  br i1 %544, label %478, label %545

545:                                              ; preds = %539
  %546 = getelementptr inbounds float, ptr %215, i64 9
  %547 = load float, ptr %546, align 4, !tbaa !34
  %548 = getelementptr inbounds float, ptr %411, i64 9
  %549 = load float, ptr %548, align 4, !tbaa !34
  %550 = fcmp contract une float %547, %549
  br i1 %550, label %478, label %551

551:                                              ; preds = %545
  %552 = getelementptr inbounds float, ptr %215, i64 10
  %553 = load float, ptr %552, align 4, !tbaa !34
  %554 = getelementptr inbounds float, ptr %411, i64 10
  %555 = load float, ptr %554, align 4, !tbaa !34
  %556 = fcmp contract une float %553, %555
  br i1 %556, label %478, label %557

557:                                              ; preds = %551
  %558 = getelementptr inbounds float, ptr %215, i64 11
  %559 = load float, ptr %558, align 4, !tbaa !34
  %560 = getelementptr inbounds float, ptr %411, i64 11
  %561 = load float, ptr %560, align 4, !tbaa !34
  %562 = fcmp contract une float %559, %561
  br i1 %562, label %478, label %563

563:                                              ; preds = %557
  %564 = getelementptr inbounds float, ptr %215, i64 12
  %565 = load float, ptr %564, align 4, !tbaa !34
  %566 = getelementptr inbounds float, ptr %411, i64 12
  %567 = load float, ptr %566, align 4, !tbaa !34
  %568 = fcmp contract une float %565, %567
  br i1 %568, label %478, label %569

569:                                              ; preds = %563
  %570 = getelementptr inbounds float, ptr %215, i64 13
  %571 = load float, ptr %570, align 4, !tbaa !34
  %572 = getelementptr inbounds float, ptr %411, i64 13
  %573 = load float, ptr %572, align 4, !tbaa !34
  %574 = fcmp contract une float %571, %573
  br i1 %574, label %478, label %575

575:                                              ; preds = %569
  %576 = getelementptr inbounds float, ptr %215, i64 14
  %577 = load float, ptr %576, align 4, !tbaa !34
  %578 = getelementptr inbounds float, ptr %411, i64 14
  %579 = load float, ptr %578, align 4, !tbaa !34
  %580 = fcmp contract une float %577, %579
  br i1 %580, label %478, label %581

581:                                              ; preds = %575
  %582 = getelementptr inbounds float, ptr %215, i64 15
  %583 = load float, ptr %582, align 4, !tbaa !34
  %584 = getelementptr inbounds float, ptr %411, i64 15
  %585 = load float, ptr %584, align 4, !tbaa !34
  %586 = fcmp contract une float %583, %585
  br i1 %586, label %478, label %587

587:                                              ; preds = %581
  %588 = getelementptr inbounds float, ptr %215, i64 16
  %589 = load float, ptr %588, align 4, !tbaa !34
  %590 = getelementptr inbounds float, ptr %411, i64 16
  %591 = load float, ptr %590, align 4, !tbaa !34
  %592 = fcmp contract une float %589, %591
  br i1 %592, label %478, label %593

593:                                              ; preds = %587
  %594 = getelementptr inbounds float, ptr %215, i64 17
  %595 = load float, ptr %594, align 4, !tbaa !34
  %596 = getelementptr inbounds float, ptr %411, i64 17
  %597 = load float, ptr %596, align 4, !tbaa !34
  %598 = fcmp contract une float %595, %597
  br i1 %598, label %478, label %599

599:                                              ; preds = %593
  %600 = getelementptr inbounds float, ptr %215, i64 18
  %601 = load float, ptr %600, align 4, !tbaa !34
  %602 = getelementptr inbounds float, ptr %411, i64 18
  %603 = load float, ptr %602, align 4, !tbaa !34
  %604 = fcmp contract une float %601, %603
  br i1 %604, label %478, label %605

605:                                              ; preds = %599
  %606 = getelementptr inbounds float, ptr %215, i64 19
  %607 = load float, ptr %606, align 4, !tbaa !34
  %608 = getelementptr inbounds float, ptr %411, i64 19
  %609 = load float, ptr %608, align 4, !tbaa !34
  %610 = fcmp contract une float %607, %609
  br i1 %610, label %478, label %611

611:                                              ; preds = %605
  %612 = getelementptr inbounds float, ptr %215, i64 20
  %613 = load float, ptr %612, align 4, !tbaa !34
  %614 = getelementptr inbounds float, ptr %411, i64 20
  %615 = load float, ptr %614, align 4, !tbaa !34
  %616 = fcmp contract une float %613, %615
  br i1 %616, label %478, label %617

617:                                              ; preds = %611
  %618 = getelementptr inbounds float, ptr %215, i64 21
  %619 = load float, ptr %618, align 4, !tbaa !34
  %620 = getelementptr inbounds float, ptr %411, i64 21
  %621 = load float, ptr %620, align 4, !tbaa !34
  %622 = fcmp contract une float %619, %621
  br i1 %622, label %478, label %623

623:                                              ; preds = %617
  %624 = getelementptr inbounds float, ptr %215, i64 22
  %625 = load float, ptr %624, align 4, !tbaa !34
  %626 = getelementptr inbounds float, ptr %411, i64 22
  %627 = load float, ptr %626, align 4, !tbaa !34
  %628 = fcmp contract une float %625, %627
  br i1 %628, label %478, label %629

629:                                              ; preds = %623
  %630 = getelementptr inbounds float, ptr %215, i64 23
  %631 = load float, ptr %630, align 4, !tbaa !34
  %632 = getelementptr inbounds float, ptr %411, i64 23
  %633 = load float, ptr %632, align 4, !tbaa !34
  %634 = fcmp contract une float %631, %633
  br i1 %634, label %478, label %635

635:                                              ; preds = %629
  %636 = getelementptr inbounds float, ptr %215, i64 24
  %637 = load float, ptr %636, align 4, !tbaa !34
  %638 = getelementptr inbounds float, ptr %411, i64 24
  %639 = load float, ptr %638, align 4, !tbaa !34
  %640 = fcmp contract une float %637, %639
  br i1 %640, label %478, label %641

641:                                              ; preds = %635
  %642 = getelementptr inbounds float, ptr %215, i64 25
  %643 = load float, ptr %642, align 4, !tbaa !34
  %644 = getelementptr inbounds float, ptr %411, i64 25
  %645 = load float, ptr %644, align 4, !tbaa !34
  %646 = fcmp contract une float %643, %645
  br i1 %646, label %478, label %647

647:                                              ; preds = %641
  %648 = getelementptr inbounds float, ptr %215, i64 26
  %649 = load float, ptr %648, align 4, !tbaa !34
  %650 = getelementptr inbounds float, ptr %411, i64 26
  %651 = load float, ptr %650, align 4, !tbaa !34
  %652 = fcmp contract une float %649, %651
  br i1 %652, label %478, label %653

653:                                              ; preds = %647
  %654 = getelementptr inbounds float, ptr %215, i64 27
  %655 = load float, ptr %654, align 4, !tbaa !34
  %656 = getelementptr inbounds float, ptr %411, i64 27
  %657 = load float, ptr %656, align 4, !tbaa !34
  %658 = fcmp contract une float %655, %657
  br i1 %658, label %478, label %659

659:                                              ; preds = %653
  %660 = getelementptr inbounds float, ptr %215, i64 28
  %661 = load float, ptr %660, align 4, !tbaa !34
  %662 = getelementptr inbounds float, ptr %411, i64 28
  %663 = load float, ptr %662, align 4, !tbaa !34
  %664 = fcmp contract une float %661, %663
  br i1 %664, label %478, label %665

665:                                              ; preds = %659
  %666 = getelementptr inbounds float, ptr %215, i64 29
  %667 = load float, ptr %666, align 4, !tbaa !34
  %668 = getelementptr inbounds float, ptr %411, i64 29
  %669 = load float, ptr %668, align 4, !tbaa !34
  %670 = fcmp contract une float %667, %669
  br i1 %670, label %478, label %671

671:                                              ; preds = %665
  %672 = getelementptr inbounds float, ptr %215, i64 30
  %673 = load float, ptr %672, align 4, !tbaa !34
  %674 = getelementptr inbounds float, ptr %411, i64 30
  %675 = load float, ptr %674, align 4, !tbaa !34
  %676 = fcmp contract une float %673, %675
  br i1 %676, label %478, label %677

677:                                              ; preds = %671
  %678 = getelementptr inbounds float, ptr %215, i64 31
  %679 = load float, ptr %678, align 4, !tbaa !34
  %680 = getelementptr inbounds float, ptr %411, i64 31
  %681 = load float, ptr %680, align 4, !tbaa !34
  %682 = fcmp contract une float %679, %681
  br i1 %682, label %478, label %683

683:                                              ; preds = %677
  %684 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.19, i64 noundef 42)
          to label %685 unwind label %716

685:                                              ; preds = %683
  %686 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !20
  %687 = getelementptr i8, ptr %686, i64 -24
  %688 = load i64, ptr %687, align 8
  %689 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %688
  %690 = getelementptr inbounds %"class.std::basic_ios", ptr %689, i64 0, i32 5
  %691 = load ptr, ptr %690, align 8, !tbaa !22
  %692 = icmp eq ptr %691, null
  br i1 %692, label %693, label %695

693:                                              ; preds = %685
  invoke void @_ZSt16__throw_bad_castv() #19
          to label %694 unwind label %716

694:                                              ; preds = %693
  unreachable

695:                                              ; preds = %685
  %696 = getelementptr inbounds %"class.std::ctype", ptr %691, i64 0, i32 8
  %697 = load i8, ptr %696, align 8, !tbaa !31
  %698 = icmp eq i8 %697, 0
  br i1 %698, label %702, label %699

699:                                              ; preds = %695
  %700 = getelementptr inbounds %"class.std::ctype", ptr %691, i64 0, i32 9, i64 10
  %701 = load i8, ptr %700, align 1, !tbaa !18
  br label %708

702:                                              ; preds = %695
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %691)
          to label %703 unwind label %716

703:                                              ; preds = %702
  %704 = load ptr, ptr %691, align 8, !tbaa !20
  %705 = getelementptr inbounds ptr, ptr %704, i64 6
  %706 = load ptr, ptr %705, align 8
  %707 = invoke noundef signext i8 %706(ptr noundef nonnull align 8 dereferenceable(570) %691, i8 noundef signext 10)
          to label %708 unwind label %716

708:                                              ; preds = %703, %699
  %709 = phi i8 [ %701, %699 ], [ %707, %703 ]
  %710 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef signext %709)
          to label %711 unwind label %716

711:                                              ; preds = %708
  %712 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %710)
          to label %713 unwind label %716

713:                                              ; preds = %711
  call void @_ZdlPv(ptr noundef nonnull %411) #18
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %20) #16
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %19) #16
  call void @_ZdlPv(ptr noundef nonnull %215) #18
  call void @_ZdlPv(ptr noundef nonnull %177) #18
  br label %714

714:                                              ; preds = %713, %136
  %715 = phi i32 [ -1, %136 ], [ 0, %713 ]
  call void @llvm.lifetime.end.p0(i64 1472, ptr nonnull %15) #16
  ret i32 %715

716:                                              ; preds = %711, %708, %703, %702, %693, %683
  %717 = landingpad { ptr, i32 }
          cleanup
  br label %718

718:                                              ; preds = %716, %496, %476, %452, %432
  %719 = phi { ptr, i32 } [ %433, %432 ], [ %453, %452 ], [ %477, %476 ], [ %497, %496 ], [ %717, %716 ]
  call void @_ZdlPv(ptr noundef nonnull %411) #18
  br label %720

720:                                              ; preds = %430, %718, %408, %383, %363, %344
  %721 = phi { ptr, i32 } [ %345, %344 ], [ %364, %363 ], [ %384, %383 ], [ %409, %408 ], [ %719, %718 ], [ %431, %430 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %20) #16
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %19) #16
  br label %722

722:                                              ; preds = %327, %720
  %723 = phi { ptr, i32 } [ %721, %720 ], [ %328, %327 ]
  call void @_ZdlPv(ptr noundef nonnull %215) #18
  br label %726

724:                                              ; preds = %194, %212, %209, %204, %203, %184, %182, %180, %176
  %725 = landingpad { ptr, i32 }
          cleanup
  br label %726

726:                                              ; preds = %724, %722, %220
  %727 = phi { ptr, i32 } [ %221, %220 ], [ %723, %722 ], [ %725, %724 ]
  call void @_ZdlPv(ptr noundef nonnull %177) #18
  br label %728

728:                                              ; preds = %726, %150
  %729 = phi { ptr, i32 } [ %151, %150 ], [ %727, %726 ]
  call void @llvm.lifetime.end.p0(i64 1472, ptr nonnull %15) #16
  resume { ptr, i32 } %729
}

declare dso_local i32 @hipGetDevicePropertiesR0600(ptr noundef, i32 noundef) local_unnamed_addr #5

; Function Attrs: inlinehint mustprogress uwtable
declare dso_local noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef) local_unnamed_addr #6

declare dso_local ptr @hipGetErrorName(i32 noundef) local_unnamed_addr #5

; Function Attrs: inlinehint mustprogress uwtable
declare dso_local noundef nonnull align 8 dereferenceable(8) ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #6

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #7

declare dso_local i32 @__gxx_personality_v0(...)

declare dso_local i32 @hipMemcpy(ptr noundef, ptr noundef, i64 noundef, i32 noundef) local_unnamed_addr #5

declare dso_local i32 @__hipPushCallConfiguration(i64, i32, i64, i32, i64 noundef, ptr noundef) local_unnamed_addr #5

declare dso_local i32 @hipFree(ptr noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctlz.i32(i32, i1 immarg) #8

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(ptr noundef) local_unnamed_addr #9

declare dso_local noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare dso_local i64 @strlen(ptr nocapture noundef) local_unnamed_addr #10

declare dso_local noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef signext) local_unnamed_addr #5

declare dso_local noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #5

; Function Attrs: noreturn
declare dso_local void @_ZSt16__throw_bad_castv() local_unnamed_addr #11

declare dso_local void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #5

declare dso_local noundef ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr noundef nonnull align 8 dereferenceable(32), ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local noundef i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4findEPKcmm(ptr noundef nonnull align 8 dereferenceable(32), ptr noundef, i64 noundef, i64 noundef) local_unnamed_addr #12

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #13

declare dso_local noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), double noundef) local_unnamed_addr #5

declare dso_local i32 @hipMalloc(ptr noundef, i64 noundef) local_unnamed_addr #5

declare dso_local i32 @__hipRegisterFunction(ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr) local_unnamed_addr

declare dso_local ptr @__hipRegisterFatBinary(ptr) local_unnamed_addr

define internal void @__hip_module_ctor() {
  %1 = load ptr, ptr @__hip_gpubin_handle_ffca4d332f194385, align 8
  %2 = icmp eq ptr %1, null
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  %4 = tail call ptr @__hipRegisterFatBinary(ptr nonnull @__hip_fatbin_wrapper)
  store ptr %4, ptr @__hip_gpubin_handle_ffca4d332f194385, align 8
  br label %5

5:                                                ; preds = %3, %0
  %6 = phi ptr [ %4, %3 ], [ %1, %0 ]
  %7 = tail call i32 @__hipRegisterFunction(ptr %6, ptr nonnull @_Z21float_to_fp8_to_floatPf26__hip_fp8_interpretation_t18__hip_saturation_tS_m, ptr nonnull @0, ptr nonnull @0, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
  %8 = tail call i32 @atexit(ptr nonnull @__hip_module_dtor)
  ret void
}

declare dso_local void @__hipUnregisterFatBinary(ptr) local_unnamed_addr

define internal void @__hip_module_dtor() {
  %1 = load ptr, ptr @__hip_gpubin_handle_ffca4d332f194385, align 8
  %2 = icmp eq ptr %1, null
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @__hipUnregisterFatBinary(ptr nonnull %1)
  store ptr null, ptr @__hip_gpubin_handle_ffca4d332f194385, align 8
  br label %4

4:                                                ; preds = %3, %0
  ret void
}

declare dso_local i32 @atexit(ptr) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #14

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #15

attributes #0 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { inlinehint mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { nobuiltin nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { mustprogress nofree nounwind willreturn memory(argmem: read) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #11 = { noreturn "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #12 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #13 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #14 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #15 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #16 = { nounwind }
attributes #17 = { noreturn nounwind }
attributes #18 = { builtin nounwind }
attributes #19 = { noreturn }
attributes #20 = { builtin allocsize(0) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{!"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.3 25012 e5bf7e55c91490b07c49d8960fa7983d864936c4)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"_ZTS26__hip_fp8_interpretation_t", !5, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"_ZTS18__hip_saturation_t", !5, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"long", !5, i64 0}
!13 = !{i32 24, i32 33}
!14 = !{!15, !4, i64 0}
!15 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !4, i64 0}
!16 = !{!17, !4, i64 0}
!17 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !15, i64 0, !12, i64 8, !5, i64 16}
!18 = !{!5, !5, i64 0}
!19 = !{!17, !12, i64 8}
!20 = !{!21, !21, i64 0}
!21 = !{!"vtable pointer", !6, i64 0}
!22 = !{!23, !4, i64 240}
!23 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !24, i64 0, !4, i64 216, !5, i64 224, !30, i64 225, !4, i64 232, !4, i64 240, !4, i64 248, !4, i64 256}
!24 = !{!"_ZTSSt8ios_base", !12, i64 8, !12, i64 16, !25, i64 24, !26, i64 28, !26, i64 32, !4, i64 40, !27, i64 48, !5, i64 64, !28, i64 192, !4, i64 200, !29, i64 208}
!25 = !{!"_ZTSSt13_Ios_Fmtflags", !5, i64 0}
!26 = !{!"_ZTSSt12_Ios_Iostate", !5, i64 0}
!27 = !{!"_ZTSNSt8ios_base6_WordsE", !4, i64 0, !12, i64 8}
!28 = !{!"int", !5, i64 0}
!29 = !{!"_ZTSSt6locale", !4, i64 0}
!30 = !{!"bool", !5, i64 0}
!31 = !{!32, !5, i64 56}
!32 = !{!"_ZTSSt5ctypeIcE", !33, i64 0, !4, i64 16, !30, i64 24, !4, i64 32, !4, i64 40, !4, i64 48, !5, i64 56, !5, i64 57, !5, i64 313, !5, i64 569}
!33 = !{!"_ZTSNSt6locale5facetE", !28, i64 8}
!34 = !{!35, !35, i64 0}
!35 = !{!"float", !5, i64 0}
!36 = !{!24, !12, i64 16}

; __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
