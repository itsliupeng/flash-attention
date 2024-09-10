import torch
from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache
from vllm_flash_attn.flash_attn_interface import flash_attn_func as vllm_flash_attn_func
from einops import rearrange


# q = torch.rand(B, N, 1, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
# # q = torch.rand(B, 1, N, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
# cache = torch.rand(num_blocks, block_size, 1, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
# cache_seqlens = torch.tensor([seqlen] * B, dtype=torch.int32, device="cuda")
# block_table = torch.randint(0, num_blocks, (B, (seqlen + block_size - 1)//block_size), dtype=torch.int32, device="cuda")

# for i in range(1):
#     out = flash_attn_with_kvcache(q, cache, cache, cache_seqlens=cache_seqlens, block_table=block_table, causal=False)
#     # print(out)
#     print(f"iter {i}")
      


# B = 8
# S = 128
N = 128
# H = 256
B, H, S = 8, 256, 128

is_causal = False
num_blocks = 32
block_size = 64



print(f"is_causal: {is_causal}")

print(">>>>> MHA")
for S in [128, 512, 1024]:
# for S in [2048, 4096, 8192]:
    for H in [128, 256]:
    # for H in [512, 576]:
        for B in [8, 16, 32, 64]:
            q = torch.rand(B, N, 1, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
            cache = torch.rand(num_blocks, block_size, 1, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
            cache_seqlens = torch.tensor([S] * B, dtype=torch.int32, device="cuda")
            block_table = torch.randint(0, num_blocks, (B, (S + block_size - 1)//block_size), dtype=torch.int32, device="cuda")
            
            expanded_block_table = block_table.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            gathered_data = cache.to(torch.float16)[expanded_block_table].to(torch.float8_e4m3fn)
            gathered_data = gathered_data.view(B, -1, 1, H)
            
            # q = torch.rand(B, 1, N, H, dtype=torch.float16, device="cuda")
            # k = torch.rand(B, S, 1, H, dtype=torch.float16, device="cuda")          
            # v = k.clone()
            

            # f_o = torch.nn.functional.scaled_dot_product_attention(q.to(torch.float16).transpose(1, 2), k.to(torch.float16).transpose(1, 2), v.to(torch.float16).transpose(1, 2), is_causal=True).transpose(1, 2)
            # if H == 576:
            #     f_o = f_o[...,:512]

            # f_o, _ = flash_attn_func(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), causal=is_causal)
            # fa2_o = vllm_flash_attn_func(q.to(torch.float16), gathered_data.to(torch.float16), gathered_data.to(torch.float16), causal=is_causal)
            fa3_o, _ = flash_attn_func(q.to(torch.float16), gathered_data.to(torch.float16), gathered_data.to(torch.float16), causal=is_causal)
            fa2_o = fa3_o
            # o, _ = flash_attn_func(q, gathered_data, gathered_data, causal=is_causal)
            o = flash_attn_with_kvcache(q, cache, cache, cache_seqlens=cache_seqlens, block_table=block_table, causal=is_causal)
                
            # torch.cuda.synchronize()

            equivalent = torch.allclose(o, fa2_o, rtol=0, atol=0.02)
            diff = abs(o.to(torch.float32) - fa2_o.to(torch.float32))
            print(f"{S} - {H} - {B}, {q.shape},  Same ? {equivalent}, {diff.max()} - {diff.sum()}")
            
            # import ipdb; ipdb.set_trace()
            
# print(">>>>> MQA")
# for S in [128, 512, 1024]:
# # for S in [2048, 4096, 8192]:
#     for H in [128, 256]:
#     # for H in [512, 576]:
#         for B in [8, 16, 32, 64]:
#             q = torch.rand(B, 1, N, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
#             k = torch.rand(B, S, 1, H, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)         
#             v = k.clone()

#             # f_o = torch.nn.functional.scaled_dot_product_attention(q.to(torch.float16).transpose(1, 2), k.to(torch.float16).transpose(1, 2), v.to(torch.float16).transpose(1, 2), is_causal=True).transpose(1, 2)
#             # if H == 576:
#             #     f_o = f_o[...,:512]
#             # f_o, _ = flash_attn_func(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), causal=is_causal)
#             f_o = vllm_flash_attn_func(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), causal=is_causal)
#             o, _ = flash_attn_func(q, k, v, causal=is_causal)

#             # torch.cuda.synchronize()

#             equivalent = torch.allclose(o, f_o, rtol=0, atol=0.02)
#             diff = abs(o.to(torch.float32) - f_o.to(torch.float32))
#             print(f"{S} - {H} - {B}, {q.shape},  Same ? {equivalent}, {diff.max()} - {diff.sum()}")