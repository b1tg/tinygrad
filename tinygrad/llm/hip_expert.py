"""Custom HIP expert GEMV — wave64, vectorized loads, multi-accumulator, x caching."""
from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.helpers import prod

def _find_raw_blocks(weight: Tensor) -> Tensor | None:
  """Walk the weight tensor's UOp chain to find the raw uchar node
  (CONTIGUOUS or realized BUFFER). Reuses existing graph node so
  the scheduler sees the same buffer dependency as the dequant path."""
  seen: set[int] = set()
  stack = [weight.uop]
  while stack:
    u = stack.pop()
    if id(u) in seen: continue
    seen.add(id(u))
    if u.op in {Ops.CONTIGUOUS, Ops.BUFFER} and u.dtype.scalar() == dtypes.uchar:
      return Tensor(u)
    stack.extend(u.src)
  return None

_HDR = """\
typedef long unsigned int size_t;
#define half _Float16
extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);
static inline __attribute__((device)) void atomicAddF(float* a, float v) {
    __asm__ volatile("global_atomic_add_f32 %0, %1, off" : : "v"(a), "v"(v) : "memory"); }
static inline __attribute__((device)) float lh(const unsigned char* p) {
    unsigned short v = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return (float)*((half*)&v); }
static inline __attribute__((device)) float silu(float x) {
    return x / (1.0f + __builtin_expf(-x)); }
#define RED32(v) do { \\
    v += __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401f)); \\
    v += __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201f)); \\
    v += __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101f)); \\
    v += __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081f)); \\
    v += __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041f)); \\
} while(0)
"""

def _make_fused_gate_up_q4k_src(H: int, D: int, K: int, NE: int, eps: float = 1e-5, fuse_norm: bool = False) -> str:
  bpr = D // 256
  es = H * bpr * 144
  RPW = 8
  if fuse_norm:
    sig = "half* out, const float* h, const half* norm_w, const int* sel, const unsigned char* wg, const unsigned char* wu"
    name = "expert_fused_norm_gate_up_q4k"
    x_load_lds = f"""
    long hoff=(long)(k/{K})*{D}L;
    float sum_sq=0.0f;
    for(int i=tid;i<{D};i+=256) {{ float v=h[hoff+i]; sum_sq+=v*v; sx[i]=v; }}
    __builtin_amdgcn_s_barrier();
    RED32(sum_sq);
    __attribute__((shared, aligned(4))) float warp_sq[{RPW}];
    if(lane==0) warp_sq[warp]=sum_sq;
    __builtin_amdgcn_s_barrier();
    sum_sq=0.0f; for(int i=0;i<{RPW};i++) sum_sq+=warp_sq[i];
    float scale=1.0f/__builtin_sqrtf(sum_sq/{D}.0f+{eps}f);
    for(int i=tid;i<{D};i+=256) sx[i]*=scale*(float)norm_w[i];
    __builtin_amdgcn_s_barrier();"""
  else:
    sig = "half* out, const float* x, const int* sel, const unsigned char* wg, const unsigned char* wu"
    name = "expert_fused_gate_up_q4k"
    x_load_lds = f"""
    long xoff=(long)(k/{K})*{D}L;
    for(int i=tid;i<{D};i+=256) sx[i]=x[xoff+i];
    __builtin_amdgcn_s_barrier();"""
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(256, 256)))
{name}({sig}) {{{{
    __attribute__((shared, aligned(16))) float sx[{D}];
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int row=__ockl_get_group_id(0)*{RPW}+warp, k=__ockl_get_group_id(1);
    int eid=sel[k];{x_load_lds}
    if(row>={H}) return;
    long roff_g=(long)eid*{es}L+(long)row*{bpr*144}L;
    long roff_u=roff_g;
    int grp=lane>>3, lpos=(lane&7)*4;
    int sub_lo=grp*2, sub_hi=grp*2+1;
    float ag=0.0f, au=0.0f;
    for(int b=0;b<{bpr};b++) {{{{
        int xb=b*256;
        float xl[4], xh[4];
        for(int j=0;j<4;j++) xl[j]=sx[xb+sub_lo*32+lpos+j];
        for(int j=0;j<4;j++) xh[j]=sx[xb+sub_hi*32+lpos+j];
        {{{{ long o=roff_g+(long)b*144L;
        float d=lh(wg+o), dm=lh(wg+o+2); const unsigned char* sc=wg+o+4;
        unsigned char svl,mvl,svh,mvh;
        if(sub_lo<4){{{{svl=sc[sub_lo]&63;mvl=sc[sub_lo+4]&63;}}}}
        else{{{{svl=(sc[sub_lo+4]&0xF)|((sc[sub_lo-4]>>6)<<4);mvl=(sc[sub_lo+4]>>4)|((sc[sub_lo]>>6)<<4);}}}}
        if(sub_hi<4){{{{svh=sc[sub_hi]&63;mvh=sc[sub_hi+4]&63;}}}}
        else{{{{svh=(sc[sub_hi+4]&0xF)|((sc[sub_hi-4]>>6)<<4);mvh=(sc[sub_hi+4]>>4)|((sc[sub_hi]>>6)<<4);}}}}
        float sl=d*(float)svl, ml=dm*(float)mvl, sh=d*(float)svh, mh=dm*(float)mvh;
        unsigned int pk=*(const unsigned int*)(wg+o+16+grp*32+lpos);
        ag+=(sl*(float)(pk&0xF)-ml)*xl[0]+(sl*(float)((pk>>8)&0xF)-ml)*xl[1]
           +(sl*(float)((pk>>16)&0xF)-ml)*xl[2]+(sl*(float)((pk>>24)&0xF)-ml)*xl[3];
        ag+=(sh*(float)((pk>>4)&0xF)-mh)*xh[0]+(sh*(float)((pk>>12)&0xF)-mh)*xh[1]
           +(sh*(float)((pk>>20)&0xF)-mh)*xh[2]+(sh*(float)((pk>>28)&0xF)-mh)*xh[3];
        }}}}
        {{{{ long o=roff_u+(long)b*144L;
        float d=lh(wu+o), dm=lh(wu+o+2); const unsigned char* sc=wu+o+4;
        unsigned char svl,mvl,svh,mvh;
        if(sub_lo<4){{{{svl=sc[sub_lo]&63;mvl=sc[sub_lo+4]&63;}}}}
        else{{{{svl=(sc[sub_lo+4]&0xF)|((sc[sub_lo-4]>>6)<<4);mvl=(sc[sub_lo+4]>>4)|((sc[sub_lo]>>6)<<4);}}}}
        if(sub_hi<4){{{{svh=sc[sub_hi]&63;mvh=sc[sub_hi+4]&63;}}}}
        else{{{{svh=(sc[sub_hi+4]&0xF)|((sc[sub_hi-4]>>6)<<4);mvh=(sc[sub_hi+4]>>4)|((sc[sub_hi]>>6)<<4);}}}}
        float sl=d*(float)svl, ml=dm*(float)mvl, sh=d*(float)svh, mh=dm*(float)mvh;
        unsigned int pk=*(const unsigned int*)(wu+o+16+grp*32+lpos);
        au+=(sl*(float)(pk&0xF)-ml)*xl[0]+(sl*(float)((pk>>8)&0xF)-ml)*xl[1]
           +(sl*(float)((pk>>16)&0xF)-ml)*xl[2]+(sl*(float)((pk>>24)&0xF)-ml)*xl[3];
        au+=(sh*(float)((pk>>4)&0xF)-mh)*xh[0]+(sh*(float)((pk>>12)&0xF)-mh)*xh[1]
           +(sh*(float)((pk>>20)&0xF)-mh)*xh[2]+(sh*(float)((pk>>28)&0xF)-mh)*xh[3];
        }}}}
    }}}}
    RED32(ag); RED32(au);
    if(lane==0) out[k*{H}+row]=(half)(silu(ag)*au);
}}}}
"""

def _q4_0_tail(quads: int, tail: int) -> str:
  body = "".join(
    f"{{ float xv=x[xoff+(b+{i})*32+lane]; long o=roff+(long)(b+{i})*18L;"
    f" float dg=lh(wg+o),du=lh(wu+o); int qg=(wg[o+2+bi]>>sh)&0xF,qu=(wu[o+2+bi]>>sh)&0xF;"
    f" ag{i}+=dg*(float)(qg-8)*xv; au{i}+=du*(float)(qu-8)*xv; }}" for i in range(tail))
  return f"\n    {{ int b={quads}*4;\n      {body}\n    }}"

def _make_fused_gate_up_q4_0_src(H: int, D: int, K: int, NE: int) -> str:
  bpr = D // 32
  es = H * bpr * 18
  quads = bpr >> 2
  tail = bpr & 3
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64, 64)))
expert_fused_gate_up_q4_0(half* out, const float* x, const int* sel,
                          const unsigned char* wg, const unsigned char* wu) {{{{
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int row=__ockl_get_group_id(0)*2+warp, k=__ockl_get_group_id(1);
    if(row>={H}) return;
    int eid=sel[k];
    long xoff=(long)(k/{K})*{D}L;
    long roff=(long)eid*{es}L+(long)row*{bpr*18}L;
    int bi=lane&0xF, sh=(lane>>4)*4;
    float ag0=0,ag1=0,ag2=0,ag3=0, au0=0,au1=0,au2=0,au3=0;
    for(int q=0;q<{quads};q++) {{{{
        int b=q<<2;
        float xv0=x[xoff+(b  )*32+lane], xv1=x[xoff+(b+1)*32+lane];
        float xv2=x[xoff+(b+2)*32+lane], xv3=x[xoff+(b+3)*32+lane];
        long o0=roff+(long)(b  )*18L, o1=roff+(long)(b+1)*18L;
        long o2=roff+(long)(b+2)*18L, o3=roff+(long)(b+3)*18L;
        #define Q4D(o,xv,ag,au) {{ \\
            float dg=lh(wg+(o)), du=lh(wu+(o)); \\
            int qg=(wg[(o)+2+bi]>>sh)&0xF, qu=(wu[(o)+2+bi]>>sh)&0xF; \\
            (ag)+=dg*(float)(qg-8)*(xv); (au)+=du*(float)(qu-8)*(xv); }}
        Q4D(o0,xv0,ag0,au0); Q4D(o1,xv1,ag1,au1);
        Q4D(o2,xv2,ag2,au2); Q4D(o3,xv3,ag3,au3);
        #undef Q4D
    }}}}
    {"" if tail == 0 else _q4_0_tail(quads, tail)}
    float ag=(ag0+ag1)+(ag2+ag3), au=(au0+au1)+(au2+au3);
    RED32(ag); RED32(au);
    if(lane==0) out[k*{H}+row]=(half)(silu(ag)*au);
}}}}
"""

def _make_expert_q4_0_src(H: int, D: int, xgroup: int, NE: int) -> str:
  bpr = D // 32
  es = H * bpr * 18
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64, 64)))
expert_gemv_q4_0(half* out, const half* x, const int* sel, const unsigned char* w) {{{{
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int row=__ockl_get_group_id(0)*2+warp, k=__ockl_get_group_id(1);
    if(row>={H}) return;
    int eid=sel[k];
    long xoff=(long)(k/{xgroup})*{D}L;
    long base=(long)eid*{es}L+(long)row*{bpr*18}L;
    int byte_idx=lane&0xF, shift=(lane>>4)*4;
    float acc=0.0f;
    for(int b=0;b<{bpr};b++) {{{{
        float xv=(float)x[xoff+b*32+lane];
        long o=base+(long)b*18L;
        float d=lh(w+o);
        int q=(w[o+2+byte_idx]>>shift)&0xF;
        acc+=d*(float)(q-8)*xv;
    }}}}
    RED32(acc);
    if(lane==0) out[k*{H}+row]=(half)acc;
}}}}
"""

def _make_expert_q8_0_src(H: int, D: int, xgroup: int, NE: int) -> str:
  bpr = D // 32
  es = H * bpr * 34
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64, 64)))
expert_gemv_q8(half* out, const half* x, const int* sel, const unsigned char* w) {{{{
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int row=__ockl_get_group_id(0)*2+warp, k=__ockl_get_group_id(1);
    if(row>={H}) return;
    long base=(long)sel[k]*{es}L+(long)row*{bpr*34}L;
    long xoff=(long)(k/{xgroup})*{D}L;
    float acc=0.0f;
    for(int b=lane; b<{bpr}; b+=32) {{{{
        long o=base+(long)b*34L; float s=lh(w+o);
        const signed char* q=(const signed char*)(w+o+2);
        float p=0.0f; for(int j=0;j<32;j++) p+=(float)x[xoff+b*32+j]*(float)q[j];
        acc+=s*p;
    }}}}
    RED32(acc);
    if(lane==0) out[k*{H}+row]=(half)acc;
}}}}
"""

def _make_mega_expert_src(D: int, H: int, K: int, NE: int, qt_gu: int, qt_d: int) -> str:
  """Full MoE expert: gate_Q4K + up_Q4K + silu*mul → LDS → down_Q8 + probs + sum. ONE dispatch.
  Grid=(ngh_out, T), workgroup=64. Each workgroup: 2 output rows, loop K experts."""
  # gate/up params (Q4_K: 256-elem blocks, 144B)
  gu_bpr = D // 256
  gu_es = H * gu_bpr * 144
  # down params (Q8_0: 32-elem blocks, 34B)

  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64, 64)))
mega_expert(half* out, const float* x, const int* sel, const float* probs,
            const unsigned char* wg, const unsigned char* wu, const unsigned char* wd) {{{{
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int out_row=__ockl_get_group_id(0)*2+warp, tok=__ockl_get_group_id(1);
    if(out_row>={D}) return;
    long xoff=(long)tok*{D}L;
    __attribute__((address_space(3))) float lds_fused[{H}];
    float acc=0.0f;
    for(int ki=0;ki<{K};ki++) {{{{
        int k=tok*{K}+ki;
        int eid=sel[k];
        
        // Phase 1: compute fused[h] = silu(gate[h]) * up[h] for all H rows
        // Each of 32 lanes handles ~H/32 rows
        for(int h=lane;h<{H};h+=32) {{{{
            long g_base=(long)eid*{gu_es}L+(long)h*{gu_bpr*144}L;
            long u_base=g_base;
            float ag=0.0f, au=0.0f;
            // Q4_K inner loop over D (8 blocks of 256)
            int grp_lane=lane>>3, lpos_lane=(lane&7)*4;
            int sub_lo=grp_lane*2, sub_hi=grp_lane*2+1;
            // For phase1 we need ALL 32 lanes to cooperate per H-row → can't parallelize H across lanes
            // WRONG: this loop structure doesn't work for Q4_K which needs 32-lane reduction
            // Need different approach: 32 lanes cooperate on ONE H-row at a time
            ag=0.0f; au=0.0f;
        }}}}
        // FIXME: Q4_K gate/up needs warp-cooperative reduction per H-row
        // Can't trivially parallelize H across lanes — each H-row needs a 32-lane dot product
        // Phase 2: down dot product
        // ...
    }}}}
}}}}
"""

def _make_quantize_q8_src(D: int, N: int) -> str:
  """Quantize fp32 x[N,D] → packed int32[N,D/4] + fp16 scales[N,D/32]. Grid=((N+1)/2,)."""
  bpr = D // 32
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64, 64)))
quantize_q8(int* out_qs, half* out_d, const float* x) {{{{
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int vec=__ockl_get_group_id(0)*2+warp;
    if(vec>={N}) return;
    long xoff=(long)vec*{D}L;
    for(int b=0;b<{bpr};b++) {{{{
        float v=x[xoff+b*32+lane];
        float av=v<0?-v:v;
        #define MR(m) {{ float o=__builtin_bit_cast(float,__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int,av),m)); if(o>av) av=o; }}
        MR(0x401f); MR(0x201f); MR(0x101f); MR(0x081f); MR(0x041f);
        #undef MR
        float d_val=av/127.0f, id=d_val>0?1.0f/d_val:0.0f;
        float qf=v*id; int qi=(int)(qf>0?qf+0.5f:qf-0.5f);
        if(qi>127) qi=127; if(qi<-128) qi=-128;
        // Pack via shift+OR: each group of 4 lanes packs into one int32
        int byte_val=qi&0xFF, sh=(lane&3)*8;
        int pk=byte_val<<sh;
        pk|=__builtin_bit_cast(int,__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int,pk),0x041f));
        pk|=__builtin_bit_cast(int,__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int,pk),0x081f));
        if((lane&3)==0) out_qs[vec*{bpr*8}+b*8+(lane>>2)]=pk;
        if(lane==0) out_d[vec*{bpr}+b]=(half)d_val;
    }}}}
}}}}
"""

def _make_expert_down_dp4a_src(out_dim: int, H: int, K: int, NE: int) -> str:
  """Down GEMV using dp4a: Q8_0 weight × Q8_1 pre-quantized input. Grid=(ngh, T*K)."""
  bpr = H // 32  # blocks per row
  es = out_dim * bpr * 34  # expert size in bytes
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64, 64)))
expert_down_dp4a(half* out, const int* x_qs, const half* x_d, const int* sel, const unsigned char* w) {{{{
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int row=__ockl_get_group_id(0)*2+warp, k=__ockl_get_group_id(1);
    if(row>={out_dim}) return;
    int eid=sel[k];
    long base=(long)eid*{es}L+(long)row*{bpr*34}L;
    float acc=0.0f;
    for(int b=lane;b<{bpr};b+=32) {{{{
        long o=base+(long)b*34L;
        float dw=lh(w+o);
        float dx=(float)x_d[k*{bpr}+b];
        // dp4a: 8 packed int32 from weight, 8 from quantized x
        const int* wq=(const int*)(w+o+2);
        int sumi=0;
        for(int i=0;i<8;i++)
            sumi=__builtin_amdgcn_sdot4(wq[i], x_qs[k*{bpr}*8+b*8+i], sumi, 0);
        acc+=dw*dx*(float)sumi;
    }}}}
    RED32(acc);
    if(lane==0) out[k*{out_dim}+row]=(half)acc;
}}}}
"""

def _make_expert_down_fused_src(out_dim: int, H: int, K: int, NE: int, qt: int) -> str:
  """Down GEMV + probs + sum over K experts. Grid=(ngh, T). Outputs half. K-sequential per workgroup."""
  bpr = H // 32
  bpb = {2: 18, 6: 22, 8: 34}.get(qt, 34)
  es = out_dim * bpr * bpb
  if qt == 2:
    inner = f"""
        float xv=(float)fused[k*{H}+b*32+lane];
        long o=base+(long)b*{bpb}L; float d=lh(w+o);
        int q=(w[o+2+byte_idx]>>shift)&0xF;
        dot+=d*(float)(q-8)*xv;"""
    loop = f"for(int b=0;b<{bpr};b++) {{{{"
    extra = "int byte_idx=lane&0xF, shift=(lane>>4)*4;"
  elif qt == 6:
    inner = f"""
        long o=base+(long)b*{bpb}L; float d=lh(w+o);
        unsigned int qh32=(unsigned int)w[o+2]|((unsigned int)w[o+3]<<8)|((unsigned int)w[o+4]<<16)|((unsigned int)w[o+5]<<24);
        float p=0.0f;
        for(int j=0;j<16;j++) {{{{ unsigned char ql=w[o+6+j];
            p+=d*(float)((ql&0xF)+((qh32>>j)&1)*16-16)*(float)fused[k*{H}+b*32+j];
            p+=d*(float)(((ql>>4)&0xF)+((qh32>>(j+16))&1)*16-16)*(float)fused[k*{H}+b*32+j+16]; }}}}
        dot+=p;"""
    loop = f"for(int b=lane;b<{bpr};b+=32) {{{{"
    extra = ""
  else:
    inner = f"""
        long o=base+(long)b*{bpb}L; float s=lh(w+o);
        float p=0.0f; for(int j=0;j<32;j++) p+=(float)fused[k*{H}+b*32+j]*(float)((signed char)w[o+2+j]);
        dot+=s*p;"""
    loop = f"for(int b=lane;b<{bpr};b+=32) {{{{"
    extra = ""
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64, 64)))
expert_down_fused(half* out, const half* fused, const int* sel, const float* probs, const unsigned char* w) {{{{
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int row=__ockl_get_group_id(0)*2+warp, tok=__ockl_get_group_id(1);
    if(row>={out_dim}) return;
    {extra}
    float acc=0.0f;
    for(int ki=0;ki<{K};ki++) {{{{
        int k=tok*{K}+ki;
        int eid=sel[k]; float prob=probs[k];
        long base=(long)eid*{es}L+(long)row*{bpr}L*{bpb}L;
        float dot=0.0f;
        {loop}
{inner}
        }}}}
        RED32(dot);
        acc+=prob*dot;
    }}}}
    if(lane==0) out[tok*{out_dim}+row]=(half)acc;
}}}}
"""

def _get_down_fused_kernel(out_dim: int, H: int, K: int, NE: int, qt: int = 8) -> tuple[str, bytes]:
  key = ("down_fused", out_dim, H, K, NE, qt)
  if key not in _kernel_cache:
    src = _make_expert_down_fused_src(out_dim, H, K, NE, qt)
    _kernel_cache[key] = (src, _compile(src))
  return _kernel_cache[key]

def _make_expert_down_atomic_src(out_dim: int, H: int, K: int, NE: int, qt: int) -> str:
  """Down GEMV + probs*scale + atomicAdd to output. Grid=(ngh, T*K), parallel over ALL expert invocations."""
  bpr = H // 32
  bpb = 18 if qt == 2 else 34
  es = out_dim * bpr * bpb
  q4_0_body = f"""
        float xv=(float)fused[k*{H}+b*32+lane];
        long o=base+(long)b*{bpb}L;
        float d=lh(w+o);
        int q=(w[o+2+byte_idx]>>shift)&0xF;
        dot+=d*(float)(q-8)*xv;"""
  q8_0_body = f"""
        long o=base+(long)b*{bpb}L; float s=lh(w+o);
        const signed char* q=(const signed char*)(w+o+2);
        float p=0.0f; for(int j=0;j<32;j++) p+=(float)fused[k*{H}+b*32+j]*(float)q[j];
        dot+=s*p;"""
  inner = q4_0_body if qt == 2 else q8_0_body
  loop = f"for(int b=0;b<{bpr};b++) {{{{" if qt == 2 else f"for(int b=lane;b<{bpr};b+=32) {{{{"
  extra_vars = "int byte_idx=lane&0xF, shift=(lane>>4)*4;" if qt == 2 else ""
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64, 64)))
expert_down_atomic(float* out, const half* fused, const int* sel, const float* probs, const unsigned char* w) {{{{
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int row=__ockl_get_group_id(0)*2+warp, k=__ockl_get_group_id(1);
    if(row>={out_dim}) return;
    int tok=k/{K};
    int eid=sel[k];
    float prob=probs[k];
    long base=(long)eid*{es}L+(long)row*{bpr}L*{bpb}L;
    {extra_vars}
    float dot=0.0f;
    {loop}
{inner}
    }}}}
    RED32(dot);
    if(lane==0) atomicAddF(&out[tok*{out_dim}+row], prob*dot);
}}}}
"""

_cc: dict = {}
def _get_compiler():
  if "c" not in _cc:
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    t = Device[Device.DEFAULT].target
    _cc["c"] = HIPCompiler(f"gfx{t[0]}{t[1]:x}{t[2]:x}")
  return _cc["c"]

def _compile(src: str):
  if src not in _cc: _cc[src] = _get_compiler().compile(src)
  return _cc[src]

def _build_expert_call(out_uop:UOp, inputs:list[UOp], params:list[UOp], grid:tuple, local:int,
                       name:str, device:str, src:str, lib:bytes, ops:int, mem:int) -> UOp:
  """Build a PROGRAM CALL for a pre-compiled expert kernel."""
  specials = [UOp.special(g, f"gidx{i}") for i, g in enumerate(grid)] + [UOp.special(local, "lidx0")]
  sink = UOp.sink(*specials, *params, arg=KernelInfo(name=name, estimates=Estimates(ops=ops, mem=mem)))
  prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=device),
            UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))
  return prg.call(out_uop, *inputs)

def _get_dp4a_kernels(out_dim: int, H: int, K: int, NE: int) -> tuple[tuple[str, bytes], tuple[str, bytes]]:
  key = ("dp4a_pair", out_dim, H, K, NE)
  if key not in _kernel_cache:
    src_q = _make_quantize_q8_src(H, K)
    src_d = _make_expert_down_dp4a_src(out_dim, H, K, NE)
    _kernel_cache[key] = ((_compile(src_q), src_q), (_compile(src_d), src_d))
  return _kernel_cache[key]

def expert_down_dp4a(fused: Tensor, sel: Tensor, probs: Tensor, raw: Tensor,
                     NE: int, out_dim: int, H: int, K: int,
                     _dp4a_cache: tuple|None = None) -> Tensor | None:
  """Quantize fused → dp4a down GEMV + probs + sum. 2 dispatches but faster inner loop."""
  if _dp4a_cache is None: return None
  (lib_q, src_q), (lib_d, src_d) = _dp4a_cache
  bpr = H // 32
  T = fused.shape[0]
  total_k = T * K
  dev = fused.device

  # Phase 1: quantize fused (T, K, H) half → int32 packed + fp16 scales
  # fused is half from gate+up output; kernel expects float → cast
  fused_f = fused.cast(dtypes.float32).contiguous()
  x_qs = Tensor.empty(total_k * bpr * 8, dtype=dtypes.int32, device=dev)
  x_d = Tensor.empty(total_k * bpr, dtype=dtypes.half, device=dev)
  p_q = [UOp.param(i, dt, sh) for i, (dt, sh) in enumerate([
    (dtypes.int32, (total_k * bpr * 8,)), (dtypes.half, (total_k * bpr,)),
    (dtypes.float32, (total_k * H,))])]
  call_q = _build_expert_call(x_qs.uop, [x_d.uop, fused_f.uop], p_q,
    grid=((total_k + 1) // 2,), local=64, name="quantize_q8", device=dev,
    src=src_q, lib=lib_q, ops=total_k * H, mem=total_k * H * 4)
  x_qs_after = x_qs.uop.after(call_q)
  x_d_after = x_d.uop.after(call_q)

  # Phase 2: dp4a down GEMV (grid per output row per expert, K-parallel)
  # Then sum probs * results. Use the non-dp4a fused approach for probs+sum.
  # Actually, dp4a kernel outputs per-expert results. Need probs+sum separately.
  # For now: output (T*K, out_dim), then probs * sum externally.
  out = Tensor.empty(total_k * out_dim, dtype=dtypes.half, device=dev)
  p_d = [UOp.param(i, dt, sh) for i, (dt, sh) in enumerate([
    (dtypes.half, (total_k * out_dim,)), (dtypes.int32, (total_k * bpr * 8,)),
    (dtypes.half, (total_k * bpr,)), (dtypes.int32, (total_k,)),
    (dtypes.uint8, raw.uop.shape)])]
  call_d = _build_expert_call(out.uop, [x_qs_after, x_d_after, sel.uop, raw.uop], p_d,
    grid=((out_dim + 1) // 2, total_k), local=64, name="expert_down_dp4a", device=dev,
    src=src_d, lib=lib_d, ops=2 * K * out_dim * H, mem=K * out_dim * bpr * 34)

  # dp4a outputs (T*K, out_dim). Need to apply probs and sum over K.
  result = Tensor(out.uop.after(call_d)).reshape(T, K, out_dim)
  return (result * probs.reshape(T, K, 1)).sum(axis=1)  # (T, out_dim)

def _make_single_proj_q4k_src(H: int, D: int, K: int, NE: int) -> str:
  bpr = D // 256
  es = H * bpr * 144
  NW = 4
  RPW = NW
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size({NW*32}, {NW*32})))
expert_single_proj_q4k(float* out, const float* x, const int* sel, const unsigned char* w) {{{{
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int row=__ockl_get_group_id(0)*{RPW}+warp, k=__ockl_get_group_id(1);
    if(row>={H}) return;
    int eid=sel[k];
    long xoff=(long)(k/{K})*{D}L;
    long roff=(long)eid*{es}L+(long)row*{bpr*144}L;
    int grp=lane>>3, lpos=(lane&7)*4;
    int sub_lo=grp*2, sub_hi=grp*2+1;
    float acc=0.0f;
    for(int b=0;b<{bpr};b++) {{{{
        int xb=b*256;
        float xl[4], xh[4];
        for(int j=0;j<4;j++) xl[j]=x[xoff+xb+sub_lo*32+lpos+j];
        for(int j=0;j<4;j++) xh[j]=x[xoff+xb+sub_hi*32+lpos+j];
        long o=roff+(long)b*144L;
        float d=lh(w+o), dm=lh(w+o+2); const unsigned char* sc=w+o+4;
        unsigned char svl,mvl,svh,mvh;
        if(sub_lo<4){{{{svl=sc[sub_lo]&63;mvl=sc[sub_lo+4]&63;}}}}
        else{{{{svl=(sc[sub_lo+4]&0xF)|((sc[sub_lo-4]>>6)<<4);mvl=(sc[sub_lo+4]>>4)|((sc[sub_lo]>>6)<<4);}}}}
        if(sub_hi<4){{{{svh=sc[sub_hi]&63;mvh=sc[sub_hi+4]&63;}}}}
        else{{{{svh=(sc[sub_hi+4]&0xF)|((sc[sub_hi-4]>>6)<<4);mvh=(sc[sub_hi+4]>>4)|((sc[sub_hi]>>6)<<4);}}}}
        float sl=d*(float)svl, ml=dm*(float)mvl, sh=d*(float)svh, mh=dm*(float)mvh;
        unsigned int pk=*(const unsigned int*)(w+o+16+grp*32+lpos);
        acc+=(sl*(float)(pk&0xF)-ml)*xl[0]+(sl*(float)((pk>>8)&0xF)-ml)*xl[1]
           +(sl*(float)((pk>>16)&0xF)-ml)*xl[2]+(sl*(float)((pk>>24)&0xF)-ml)*xl[3];
        acc+=(sh*(float)((pk>>4)&0xF)-mh)*xh[0]+(sh*(float)((pk>>12)&0xF)-mh)*xh[1]
           +(sh*(float)((pk>>20)&0xF)-mh)*xh[2]+(sh*(float)((pk>>28)&0xF)-mh)*xh[3];
    }}}}
    RED32(acc);
    if(lane==0) out[k*{H}+row]=acc;
}}}}
"""

def expert_single_proj(x: Tensor, sel: Tensor, raw_w: Tensor, NE: int, H: int, D: int, K: int, qt: int = 12,
                       _src_lib: tuple|None = None) -> Tensor | None:
  if qt != 12: return None
  key = ("single_proj", H, D, K, NE, qt)
  if _src_lib is None:
    if key not in _kernel_cache:
      src = _make_single_proj_q4k_src(H, D, K, NE)
      _kernel_cache[key] = (src, _compile(src))
    _src_lib = _kernel_cache[key]
  src, lib = _src_lib
  rpw = 8
  ngh = (H + rpw - 1) // rpw
  bpr = D // 256
  T = prod(x.shape) // D
  total_k = T * K
  ops, mem = 2 * K * H * D, K * H * bpr * 144 + D * 4 + K * 4 + K * H * 4
  out = Tensor.empty(total_k * H, dtype=dtypes.float32, device=x.device)
  # pass x/sel WITHOUT contiguous — let scheduler handle materialization like it does for matmul
  srcs = (out.uop, x.uop, sel.uop, raw_w.uop)
  placeholders = [UOp.placeholder_like(s, slot=i) for i, s in enumerate(srcs)]
  o, xf, sf, wf = placeholders
  sink = UOp.sink(UOp.special(ngh, "gidx0"), UOp.special(total_k, "gidx1"), UOp.special(128, "lidx0"),
                  o, xf, sf, wf, arg=KernelInfo(name="expert_single_proj", estimates=Estimates(ops=ops, mem=mem)))
  program = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=x.device),
               UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))
  kernel = program.call(*srcs)
  out_after = Tensor(out.uop.after(kernel))
  return out_after.reshape(T, K, H)

def _make_mega_q4k_src(H: int, D: int, K: int, NE: int, qt_down: int = 8) -> str:
  """Mega-kernel: gate(Q4_K)+up(Q4_K)+silu+down(Q8_0 or Q5_0)+probs in ONE kernel."""
  bpr_gu = D // 256
  es_gu = H * bpr_gu * 144
  dn_blk = {8: 34, 6: 22}[qt_down]  # Q8_0=34, Q5_0=22 bytes per 32-element block
  H_PAD = ((H + 31) // 32) * 32
  bpr_dn = H_PAD // 32
  es_dn = D * bpr_dn * dn_blk
  NW = 4
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size({NW*32}, {NW*32})))
expert_mega_q4k_q8(float* out, const float* x, const int* sel, const float* probs,
                   const unsigned char* wg, const unsigned char* wu, const unsigned char* wd) {{{{
    __attribute__((shared, aligned(16))) float sx[{D}];
    __attribute__((shared, aligned(16))) float fused[{H}];
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int k=__ockl_get_group_id(0);
    int eid=sel[k];
    float prob=probs[k];
    
    long xoff=(long)(k/{K})*{D}L;
    for(int i=tid;i<{D};i+={NW*32}) sx[i]=x[xoff+i];
    __builtin_amdgcn_s_barrier();
    /* Phase 1: gate+up+silu → fused[H] in LDS (Q4_K) */
    int grp=lane>>3, lpos=(lane&7)*4;
    int sub_lo=grp*2, sub_hi=grp*2+1;
    for(int row=warp;row<{H};row+={NW}) {{{{
        long roff=(long)eid*{es_gu}L+(long)row*{bpr_gu*144}L;
        float ag=0.0f, au=0.0f;
        for(int b=0;b<{bpr_gu};b++) {{{{
            int xb=b*256;
            float xl[4], xh[4];
            for(int j=0;j<4;j++) xl[j]=sx[xb+sub_lo*32+lpos+j];
            for(int j=0;j<4;j++) xh[j]=sx[xb+sub_hi*32+lpos+j];
            {{{{ long o=roff+(long)b*144L;
            float d=lh(wg+o), dm=lh(wg+o+2); const unsigned char* sc=wg+o+4;
            unsigned char svl,mvl,svh,mvh;
            if(sub_lo<4){{{{svl=sc[sub_lo]&63;mvl=sc[sub_lo+4]&63;}}}}
            else{{{{svl=(sc[sub_lo+4]&0xF)|((sc[sub_lo-4]>>6)<<4);mvl=(sc[sub_lo+4]>>4)|((sc[sub_lo]>>6)<<4);}}}}
            if(sub_hi<4){{{{svh=sc[sub_hi]&63;mvh=sc[sub_hi+4]&63;}}}}
            else{{{{svh=(sc[sub_hi+4]&0xF)|((sc[sub_hi-4]>>6)<<4);mvh=(sc[sub_hi+4]>>4)|((sc[sub_hi]>>6)<<4);}}}}
            float sl=d*(float)svl, ml=dm*(float)mvl, sh=d*(float)svh, mh=dm*(float)mvh;
            unsigned int pk=*(const unsigned int*)(wg+o+16+grp*32+lpos);
            ag+=(sl*(float)(pk&0xF)-ml)*xl[0]+(sl*(float)((pk>>8)&0xF)-ml)*xl[1]
               +(sl*(float)((pk>>16)&0xF)-ml)*xl[2]+(sl*(float)((pk>>24)&0xF)-ml)*xl[3];
            ag+=(sh*(float)((pk>>4)&0xF)-mh)*xh[0]+(sh*(float)((pk>>12)&0xF)-mh)*xh[1]
               +(sh*(float)((pk>>20)&0xF)-mh)*xh[2]+(sh*(float)((pk>>28)&0xF)-mh)*xh[3]; }}}}
            {{{{ long o=(long)eid*{es_gu}L+(long)row*{bpr_gu*144}L+(long)b*144L;
            float d=lh(wu+o), dm=lh(wu+o+2); const unsigned char* sc=wu+o+4;
            unsigned char svl,mvl,svh,mvh;
            if(sub_lo<4){{{{svl=sc[sub_lo]&63;mvl=sc[sub_lo+4]&63;}}}}
            else{{{{svl=(sc[sub_lo+4]&0xF)|((sc[sub_lo-4]>>6)<<4);mvl=(sc[sub_lo+4]>>4)|((sc[sub_lo]>>6)<<4);}}}}
            if(sub_hi<4){{{{svh=sc[sub_hi]&63;mvh=sc[sub_hi+4]&63;}}}}
            else{{{{svh=(sc[sub_hi+4]&0xF)|((sc[sub_hi-4]>>6)<<4);mvh=(sc[sub_hi+4]>>4)|((sc[sub_hi]>>6)<<4);}}}}
            float sl=d*(float)svl, ml=dm*(float)mvl, sh=d*(float)svh, mh=dm*(float)mvh;
            unsigned int pk=*(const unsigned int*)(wu+o+16+grp*32+lpos);
            au+=(sl*(float)(pk&0xF)-ml)*xl[0]+(sl*(float)((pk>>8)&0xF)-ml)*xl[1]
               +(sl*(float)((pk>>16)&0xF)-ml)*xl[2]+(sl*(float)((pk>>24)&0xF)-ml)*xl[3];
            au+=(sh*(float)((pk>>4)&0xF)-mh)*xh[0]+(sh*(float)((pk>>12)&0xF)-mh)*xh[1]
               +(sh*(float)((pk>>20)&0xF)-mh)*xh[2]+(sh*(float)((pk>>28)&0xF)-mh)*xh[3]; }}}}
        }}}}
        RED32(ag); RED32(au);
        if(lane==0) fused[row]=silu(ag)*au;
    }}}}
    __builtin_amdgcn_s_barrier();
    /* Phase 2: down + probs */
    for(int col=warp;col<{D};col+={NW}) {{{{
        long roff_d=(long)eid*{es_dn}L+(long)col*{bpr_dn*dn_blk}L;
        float acc=0.0f;
        for(int b=0;b<{bpr_dn};b++) {{{{
            int hb=b*32;
            long o=roff_d+(long)b*{dn_blk}L;
            float d=lh(wd+o);
            int idx=hb+lane;
            float fv=(idx<{H})?fused[idx]:0.0f;
{f'            acc+=d*(float)((signed char)wd[o+2+lane])*fv;' if qt_down == 8 else f"""            int qs_idx=(lane<16)?lane:(lane-16);
            unsigned char ql=wd[o+6+qs_idx]; int nib=(lane<16)?(ql&0xF):((ql>>4)&0xF);
            unsigned char qh_byte=wd[o+2+(lane>>3)]; int hi=(qh_byte>>(lane&7))&1;
            acc+=d*(float)(nib+hi*16-16)*fv;"""}
        }}}}
        RED32(acc);
        if(lane==0) {{ if(k==0) out[col]=acc*prob; else atomicAddF(out+col, acc*prob); }}
    }}}}
}}}}
"""

def expert_mega(x: Tensor, sel: Tensor, probs: Tensor, raw_g: Tensor, raw_u: Tensor, raw_d: Tensor,
                NE: int, H: int, D: int, K: int, qt_down: int = 8, _src_lib: tuple|None = None) -> Tensor | None:
  key = ("mega", H, D, K, NE, qt_down)
  if _src_lib is None:
    if key not in _kernel_cache:
      s = _make_mega_q4k_src(H, D, K, NE, qt_down)
      _kernel_cache[key] = (s, _compile(s))
    _src_lib = _kernel_cache[key]
  src, lib = _src_lib
  NW = 4
  dn_blk = {8: 34, 6: 22}[qt_down]
  bpr_gu = D // 256
  H_PAD = ((H + 31) // 32) * 32
  bpr_dn = H_PAD // 32
  ops = 2 * K * H * D * 2 + 2 * K * D * H
  mem = K * H * bpr_gu * 144 * 2 + K * D * bpr_dn * dn_blk + K * D * 4
  out = Tensor.empty(D, dtype=dtypes.float32, device=x.device)
  def fxn(o, xf, sf, pf, wgf, wuf, wdf):
    sink = UOp.sink(UOp.special(K, "gidx0"), UOp.special(NW*32, "lidx0"),
                    o, xf, sf, pf, wgf, wuf, wdf,
                    arg=KernelInfo(name="expert_mega", estimates=Estimates(ops=ops, mem=mem)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=x.device),
               UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))
  out, *_ = Tensor.custom_kernel(out, x, sel, probs, raw_g, raw_u, raw_d, fxn=fxn)
  return out

_kernel_cache: dict[tuple, tuple[str, bytes]] = {}
def _get_fused_gu_kernel(H: int, D: int, K: int, NE: int, qt: int = 12, eps: float = 1e-5, fuse_norm: bool = False) -> tuple[str, bytes]:
  key = ("fused_gu", H, D, K, NE, qt, fuse_norm)
  if key not in _kernel_cache:
    if qt == 2:
      src = _make_fused_gate_up_q4_0_src(H, D, K, NE)
    else:
      src = _make_fused_gate_up_q4k_src(H, D, K, NE, eps, fuse_norm)
    _kernel_cache[key] = (src, _compile(src))
  return _kernel_cache[key]

def _get_q8_kernel(H: int, D: int, xgroup: int, NE: int, qt: int = 8) -> tuple[str, bytes]:
  key = ("gemv", H, D, xgroup, NE, qt)
  if key not in _kernel_cache:
    src = _make_expert_q4_0_src(H, D, xgroup, NE) if qt == 2 else _make_expert_q8_0_src(H, D, xgroup, NE)
    _kernel_cache[key] = (src, _compile(src))
  return _kernel_cache[key]

def _get_down_atomic_kernel(out_dim: int, H: int, K: int, NE: int, qt: int = 8) -> tuple[str, bytes]:
  key = ("down_atomic", out_dim, H, K, NE, qt)
  if key not in _kernel_cache:
    src = _make_expert_down_atomic_src(out_dim, H, K, NE, qt)
    _kernel_cache[key] = (src, _compile(src))
  return _kernel_cache[key]

def pre_realize_weights():
  from tinygrad.llm.gguf import expert_raw_weights
  to_realize = [t.contiguous() for t, qt, _ in expert_raw_weights.values() if qt in {2, 8, 12}]
  if to_realize: Tensor.realize(*to_realize)
  for name in list(expert_raw_weights):
    t, qt, ne = expert_raw_weights[name]
    if qt in {2, 8, 12}: expert_raw_weights[name] = (t.contiguous(), qt, ne)

def expert_fused_gate_up(x: Tensor, sel: Tensor, raw_g: Tensor, raw_u: Tensor,
                         NE: int, H: int, D: int, K: int = 6, qt: int = 12,
                         _src_lib: tuple|None = None, norm_w: Tensor|None = None) -> Tensor | None:
  if qt not in {2, 12}: return None
  src, lib = _src_lib or _get_fused_gu_kernel(H, D, K, NE, qt, fuse_norm=norm_w is not None)
  rpw = 4 if qt == 12 else 2
  ngh = (H + rpw - 1) // rpw
  bpb = 18 if qt == 2 else 144
  bpr = D // (32 if qt == 2 else 256)
  T = prod(x.shape) // D
  total_k = T * K
  ops, mem = 4 * K * H * D, 2 * K * H * bpr * bpb
  out = Tensor.empty(total_k * H, dtype=dtypes.half, device=x.device)
  wg_size = 256 if qt == 12 else 64
  if norm_w is not None:
    def fxn(o, hf, nw, sf, wg, wu):
      sink = UOp.sink(UOp.special(ngh, "gidx0"), UOp.special(total_k, "gidx1"), UOp.special(wg_size, "lidx0"),
                      o, hf, nw, sf, wg, wu, arg=KernelInfo(name="expert_norm_gu", estimates=Estimates(ops=ops, mem=mem)))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=x.device),
                 UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))
    out, *_ = Tensor.custom_kernel(out, x, norm_w, sel, raw_g, raw_u, fxn=fxn)
  else:
    def fxn(o, xf, sf, wg, wu):
      sink = UOp.sink(UOp.special(ngh, "gidx0"), UOp.special(total_k, "gidx1"), UOp.special(wg_size, "lidx0"),
                      o, xf, sf, wg, wu, arg=KernelInfo(name="expert_fused_gu", estimates=Estimates(ops=ops, mem=mem)))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=x.device),
                 UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))
    out, *_ = Tensor.custom_kernel(out, x, sel, raw_g, raw_u, fxn=fxn)
  return out

def expert_down_fused(fused: Tensor, sel: Tensor, probs: Tensor, raw: Tensor,
                      NE: int, out_dim: int, H: int, K: int, qt: int = 8,
                      _src_lib: tuple|None = None) -> Tensor | None:
  """Down GEMV + probs + sum over K experts. Grid=(ngh, T). Output: (T, out_dim) half."""
  if qt not in {2, 6, 8, 12}: return None
  src, lib = _src_lib or _get_down_fused_kernel(out_dim, H, K, NE, qt)
  ngh = (out_dim + 1) // 2
  bpr = H // 32
  bpb = {2: 18, 6: 22, 8: 34}.get(qt, 34)
  T = prod(fused.shape) // (K * H)
  ops, mem = 2 * K * out_dim * H, K * out_dim * bpr * bpb
  out = Tensor.empty(T * out_dim, dtype=dtypes.half, device=fused.device)
  def fxn(o, ff, sf, pf, w):
    sink = UOp.sink(UOp.special(ngh, "gidx0"), UOp.special(T, "gidx1"), UOp.special(64, "lidx0"),
                    o, ff, sf, pf, w, arg=KernelInfo(name="expert_down_fused", estimates=Estimates(ops=ops, mem=mem)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=fused.device),
               UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))
  out, *_ = Tensor.custom_kernel(out, fused, sel, probs, raw, fxn=fxn)
  return out.reshape(T, out_dim)

def expert_gemv_q8(x: Tensor, sel: Tensor, wname: str, NE: int, H: int, D: int,
                   K: int, xgroup: int = 0, _src_lib: tuple|None = None) -> Tensor | None:
  from tinygrad.llm.gguf import expert_raw_weights
  if wname not in expert_raw_weights: return None
  raw, qt, _ = expert_raw_weights[wname]
  if qt not in {2, 8}: return None

  if xgroup == 0: xgroup = K
  src, lib = _src_lib or _get_q8_kernel(H, D, xgroup, NE, qt)
  ngh = (H + 1) // 2
  bpr = D // 32
  bpb = 18 if qt == 2 else 34

  T = x.shape[0]
  total_k = T * K
  ops, mem = 2 * K * H * D, K * H * bpr * bpb

  xf = x.contiguous()
  sf = sel.reshape(-1).contiguous()
  out = Tensor.empty(total_k * H, dtype=dtypes.half, device=x.device)

  p_out = UOp.param(0, dtypes.half, (total_k * H,))
  p_xf = UOp.param(1, x.dtype, (T * D,))
  p_sf = UOp.param(2, dtypes.int32, (total_k,))
  p_w = UOp.param(3, dtypes.uint8, raw.uop.shape)

  call = _build_expert_call(
    out.uop, [xf.uop, sf.uop, raw.uop],
    [p_out, p_xf, p_sf, p_w],
    grid=(ngh, total_k), local=64,
    name="expert_gemv_q8", device=x.device, src=src, lib=lib, ops=ops, mem=mem)

  return Tensor(out.uop.after(call)).reshape(T, K, H)

# ---- Q8_0 nn.Linear custom GEMV (no expert indexing) ----

def _make_linear_q8_0_src(H: int, D: int) -> str:
  bpr = D // 32
  return _HDR + f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64, 64)))
linear_q8_gemv(float* out, const float* x, const unsigned char* w, const int toks) {{{{
    int tid=__ockl_get_local_id(0), warp=tid>>5, lane=tid&31;
    int row=__ockl_get_group_id(0)*2+warp, tok=__ockl_get_group_id(1);
    if(row>={H}) return;
    long base=(long)row*{bpr*34}L;
    long xoff=(long)tok*{D}L;
    float acc=0.0f;
    for(int b=lane;b<{bpr};b+=32) {{{{
        long o=base+(long)b*34L; float s=lh(w+o);
        const signed char* q=(const signed char*)(w+o+2);
        float p=0.0f; for(int j=0;j<32;j++) p+=x[xoff+b*32+j]*(float)q[j];
        acc+=s*p;
    }}}}
    RED32(acc);
    if(lane==0) out[tok*{H}+row]=acc;
}}}}
"""

def _get_linear_q8_kernel(H: int, D: int) -> tuple[str, bytes]:
  key = ("linear_q8", H, D)
  if key not in _kernel_cache:
    src = _make_linear_q8_0_src(H, D)
    _kernel_cache[key] = (src, _compile(src))
  return _kernel_cache[key]

def linear_q8_gemv(x: Tensor, raw_w: Tensor, H: int, D: int, _src_lib: tuple|None = None) -> Tensor:
  src, lib = _src_lib or _get_linear_q8_kernel(H, D)
  ngh = (H + 1) // 2
  bpr = D // 32
  T = prod(x.shape) // D
  ops, mem = 2 * T * H * D, H * bpr * 34 + T * D * 4
  out = Tensor.zeros(T * H, dtype=dtypes.float32, device=x.device)
  def fxn(o, xf, wf):
    sink = UOp.sink(UOp.special(ngh, "gidx0"), UOp.special(T, "gidx1"), UOp.special(64, "lidx0"),
                    o, xf, wf, arg=KernelInfo(name="linear_q8", estimates=Estimates(ops=ops, mem=mem)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=x.device),
               UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))
  out, *_ = Tensor.custom_kernel(out, x, raw_w, fxn=fxn)
  return out
