# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn_interface import flash_attn_func

try:
    from triton_fused_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

cudnn = None

def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def time_fwd(func, *args, **kwargs):
    time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean


torch.manual_seed(0)

repeats = 30
device = 'cuda'
# dtype = torch.float16
dtype = torch.float8_e4m3fn

bs = 1024
bs_seqlen_vals = [(bs, 512), (bs, 1024), (bs, 2048), (bs, 4224), (bs, 8448), (bs, 8448 * 2), (bs, 8448 * 4)]
# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 8192 * 2)]
# bs_seqlen_vals = [(4, 4096), (2, 8192), (1, 8192 * 2), (4, 4224), (2, 8448), (1, 8448 * 2)]
# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048)]
causal_vals = [False]
# causal_vals = [False]
headdim_vals = [256]
# dim = 2048
# dim = 256
dropout_p = 0.0

methods = (["Flash3"]        
        # + (["Triton"] if attention_triton is not None else [])
        #    + (["xformers.c"] if xops is not None else [])
        #    + (["xformers.f"] if xops is not None else [])
           )

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            torch.cuda.empty_cache()
            config = (causal, headdim, batch_size, seqlen)
            # nheads = dim // headdim
            nheads = 128
            # q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.float16, requires_grad=False) for _ in range(3)]
            
            q = torch.randn(batch_size, 1, nheads, headdim, device=device, dtype=torch.float16, requires_grad=False)
            k = torch.randn(batch_size, seqlen, 1, headdim, device=device, dtype=torch.float16, requires_grad=False)
            v = torch.randn(batch_size, seqlen, 1, headdim, device=device, dtype=torch.float16, requires_grad=False)
            # qkv = torch.stack([q, k, v], dim=2)
            # qkv = qkv.to(torch.float16)
            # f = time_fwd(attention_pytorch, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False)
            # time_f[config, "Pytorch"] = f
            # res_baseline = attention_pytorch(qkv, dropout_p, causal=causal)

            if attention_triton is not None:
                q_transposed = q.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
                k_transposed = k.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
                v_transposed = v.transpose(1, 2).contiguous().permute(0, 1, 3, 2).to(torch.float8_e4m3fn)
                scale = 1 / math.sqrt(headdim)
                f = time_fwd(
                    attention_triton, q_transposed, k_transposed, v_transposed,
                    causal, scale, repeats=5, verbose=False, desc='Triton'
                )
                f = time_fwd(
                    attention_triton, q_transposed, k_transposed, v_transposed,
                    causal, scale, repeats=repeats, verbose=False, desc='Triton'
                )
                time_f[config, "Triton"] = f
                res = attention_triton(
                    q_transposed, k_transposed, v_transposed.permute(0, 1, 3, 2),
                    causal, scale
                ).half().transpose(1, 2)
                torch.testing.assert_close(res, res_baseline, atol=0.5, rtol=0.5)

            # out = torch.empty_like(q)
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)                        
            f = time_fwd(flash_attn_func, q, k, v, causal=causal, repeats=repeats, verbose=False)

            # res = flash_attn_func(q, k, v, causal=causal)
            # torch.testing.assert_close(res.half(), res_baseline, atol=0.05, rtol=0.05)

            time_f[config, "Flash3"] = f

            if cudnn is not None:
                qkv_fp8 = qkv.to(dtype)
                time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                f = time_fwd(
                    cudnn_spda_setup(
                        qkv_fp8, seqlen, seqlen,
                        causal=causal
                    ),
                    repeats=repeats, verbose=False
                )
                time_f[config, "cuDNN"] = f
                # res, amax_o = cudnn_spda_setup(
                #     qkv_fp8, seqlen, seqlen,
                #     causal=causal
                # )()
                # res = res.half()
                # TODO: CUDNN has numerics issues when
                # num_heads=16, dim=128, seq_len=1024, batch_size=2
                # or larger sizes.
                # res_cpu = res.cpu().reshape(-1)
                # res_baseline_cpu = res_baseline.cpu().reshape(-1)
                # print(amax_o)
                # print(res)
                # print(res_baseline)
                # for i in range(len(res_cpu)):
                #     item = res_cpu[i]
                #     item_baseline = res_baseline_cpu[i]
                #     if abs(item - item_baseline) > 0.5:
                #         print(i)
                #         print(item)
                #         print(item_baseline)
                # torch.testing.assert_close(res, res_baseline, atol=0.05, rtol=0.05)

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            for method in methods:
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                    time_f[config, method]
                )
                #print (time_f[config,method])
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {time_f[config, method] * 1e3} ms, "
                )


# with open('flash3_attn_time.plk', 'wb') as fp:
#     pickle.dump((time_f, time_b, time_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
