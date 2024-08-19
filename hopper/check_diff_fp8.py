import torch
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from vllm_flash_attn.flash_attn_interface import flash_attn_func as vllm_flash_attn_func
from einops import rearrange

def generate_varlen_qkv(q, k):
    batch_size, seqlen_q, _, _ = q.shape
    _, seqlen_k, _, _ = k.shape
    
    q_unpad = rearrange(q, "b s h d -> (b s) h d")
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device)
    max_seqlen_q = seqlen_q
    output_pad_fn = lambda output_unpad: rearrange(output_unpad, "(b s) h d -> b s h d", b=batch_size)
    k_unpad = rearrange(k, "b s h d -> (b s) h d")
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device)
    max_seqlen_k = seqlen_k
    return q_unpad, k_unpad, output_pad_fn, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
    


# B = 8
# S = 128
N = 32
# H = 256
B, H, S = 8, 256, 128

is_causal = False
print(f"is_causal: {is_causal}")

print(">>>>> MHA")
for S in [128, 512, 1024]:
# for S in [2048, 4096, 8192]:
    for H in [128, 256]:
    # for H in [512, 576]:
        for B in [8, 16, 32, 64]:
            q = torch.rand(B, S, N, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
            k = torch.rand(B, S, N, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
            # q = torch.rand(B, 1, N, H, dtype=torch.float16, device="cuda")
            # k = torch.rand(B, S, 1, H, dtype=torch.float16, device="cuda")          
            v = k.clone()

            # f_o = torch.nn.functional.scaled_dot_product_attention(q.to(torch.float16).transpose(1, 2), k.to(torch.float16).transpose(1, 2), v.to(torch.float16).transpose(1, 2), is_causal=True).transpose(1, 2)
            # if H == 576:
            #     f_o = f_o[...,:512]

            # f_o, _ = flash_attn_func(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), causal=is_causal)
            f_o = vllm_flash_attn_func(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), causal=is_causal)
            # o, _ = flash_attn_func(q, k, v, causal=is_causal)
            q_unpad, k_unpad, output_pad_fn, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = generate_varlen_qkv(q, k)
            o, _ = flash_attn_varlen_func(q_unpad, k_unpad, k_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=is_causal)
            o = output_pad_fn(o)
                
            # torch.cuda.synchronize()

            equivalent = torch.allclose(o, f_o, rtol=0, atol=0.02)
            diff = abs(o.to(torch.float32) - f_o.to(torch.float32))
            print(f"{S} - {H} - {B}, {q.shape},  Same ? {equivalent}, {diff.max()} - {diff.sum()}")
            
print(">>>>> MQA")
for S in [128, 512, 1024]:
# for S in [2048, 4096, 8192]:
    for H in [128, 256]:
    # for H in [512, 576]:
        for B in [8, 16, 32, 64]:
            q = torch.rand(B, 1, N, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
            k = torch.rand(B, S, 1, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)         
            v = k.clone()

            # f_o = torch.nn.functional.scaled_dot_product_attention(q.to(torch.float16).transpose(1, 2), k.to(torch.float16).transpose(1, 2), v.to(torch.float16).transpose(1, 2), is_causal=True).transpose(1, 2)
            # if H == 576:
            #     f_o = f_o[...,:512]
            # f_o, _ = flash_attn_func(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), causal=is_causal)
            f_o = vllm_flash_attn_func(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), causal=is_causal)
            # o, _ = flash_attn_func(q, k, v, causal=is_causal)
            q_unpad, k_unpad, output_pad_fn, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = generate_varlen_qkv(q, k)
            o, _ = flash_attn_varlen_func(q_unpad, k_unpad, k_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=is_causal)
            o = output_pad_fn(o)

            # torch.cuda.synchronize()

            equivalent = torch.allclose(o, f_o, rtol=0, atol=0.02)
            diff = abs(o.to(torch.float32) - f_o.to(torch.float32))
            print(f"{S} - {H} - {B}, {q.shape},  Same ? {equivalent}, {diff.max()} - {diff.sum()}")