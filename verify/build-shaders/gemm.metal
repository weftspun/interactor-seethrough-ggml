#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 35 "gemm.slang"
struct GlobalParams_0
{
    uint M_0;
    uint N_0;
    uint K_0;
};


#line 27
struct KernelContext_0
{
    float device* A_0;
    float device* B_0;
    float device* C_0;
    GlobalParams_0 constant* globalParams_0;
    array<half, int(128)> threadgroup* sA_0;
    array<half, int(128)> threadgroup* sB_0;
};


#line 18
[[kernel]] void gemm(uint3 dtid_0 [[thread_position_in_grid]], uint3 gtid_0 [[thread_position_in_threadgroup]], float device* A_1 [[buffer(1)]], float device* B_1 [[buffer(2)]], float device* C_1 [[buffer(3)]], GlobalParams_0 constant* globalParams_1 [[buffer(0)]])
{

#line 18
    thread KernelContext_0 kernelContext_0;

#line 18
    (&kernelContext_0)->A_0 = A_1;

#line 18
    (&kernelContext_0)->B_0 = B_1;

#line 18
    (&kernelContext_0)->C_0 = C_1;

#line 18
    (&kernelContext_0)->globalParams_0 = globalParams_1;

#line 18
    threadgroup array<half, int(128)> sA_1;

#line 18
    (&kernelContext_0)->sA_0 = &sA_1;

#line 18
    threadgroup array<half, int(128)> sB_1;

#line 18
    (&kernelContext_0)->sB_0 = &sB_1;

#line 24
    uint _S1 = (globalParams_1->K_0 + 7U) / 8U;

#line 24
    float sum_0 = 0.0f;

#line 24
    uint kt_0 = 0U;
    for(;;)
    {

#line 25
        if(kt_0 < _S1)
        {
        }
        else
        {

#line 25
            break;
        }

#line 26
        uint kk_0 = kt_0 * 8U;
        uint _S2 = gtid_0[0U] * 8U;

#line 27
        uint _S3 = _S2 + gtid_0[1U];

#line 27
        (*(&kernelContext_0)->sA_0)[_S3] = half((&kernelContext_0)->A_0[dtid_0[0U] * globalParams_1->K_0 + (kk_0 + gtid_0[1U])]);
        (*(&kernelContext_0)->sB_0)[_S3] = half((&kernelContext_0)->B_0[(kk_0 + gtid_0[0U]) * (&kernelContext_0)->globalParams_0->N_0 + dtid_0[1U]]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 29
        uint k_0 = 0U;
        for(;;)
        {

#line 30
            if(k_0 < 8U)
            {
            }
            else
            {

#line 30
                break;
            }

#line 31
            float _S4 = sum_0 + float((*(&kernelContext_0)->sA_0)[_S2 + k_0]) * float((*(&kernelContext_0)->sB_0)[k_0 * 8U + gtid_0[1U]]);

#line 30
            uint k_1 = k_0 + 1U;

#line 30
            sum_0 = _S4;

#line 30
            k_0 = k_1;

#line 30
        }


        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 25
        kt_0 = kt_0 + 1U;

#line 25
    }

#line 35
    *((&kernelContext_0)->C_0+(dtid_0[0U] * (&kernelContext_0)->globalParams_0->N_0 + dtid_0[1U])) = sum_0;
    return;
}

