import torch
from flash_attn_interface import flash_attn_func
from vllm_flash_attn.flash_attn_interface import flash_attn_func as vllm_flash_attn_func

# B = 8
# S = 128
N = 32
# H = 256

for S in [128, 512, 1024]:
    for H in [128, 256]:
        for B in [8, 16, 32, 64]:
            # q = torch.rand(B, S, N, H, dtype=torch.float16, device="cuda")
            # k = torch.rand(B, S, N, H, dtype=torch.float16, device="cuda")
            q = torch.rand(B, 1, N, H, dtype=torch.float16, device="cuda")
            k = torch.rand(B, S, 1, H, dtype=torch.float16, device="cuda")          
            v = k.clone()

            # f_o = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            # f_o = f_o.transpose(1, 2)

            f_o = vllm_flash_attn_func(q, k, v, causal=True)
            o, _ = flash_attn_func(q, k, v, causal=True)

            # torch.cuda.synchronize()

            equivalent = torch.allclose(o, f_o, rtol=1e-2, atol=1e-3)
            diff = abs(o - f_o)
            print(f"{S} - {H} - {B}, {q.shape},  Same ? {equivalent}, {diff.max()} - {diff.sum()}")