#!/usr/bin/env python3

import torch
from memory_efficient_attention_pytorch import Attention
from memory_efficient_attention_pytorch.flash_attention import FlashAttention, FlashAttentionFunction


attn_kwargs = dict(
    dim=512,
    dim_head=64,
    heads=8,
    q_bucket_size=64,
    k_bucket_size=32,
    causal=False)

attn = Attention(**attn_kwargs)
flash_attn = FlashAttention(**attn_kwargs)

flash_attn.to_q = attn.to_q
flash_attn.to_kv = attn.to_kv
flash_attn.to_out = attn.to_out

x = torch.randn(2, 2048, 512)
# mask = torch.ones(2, 2048).bool()
mask = None

mem_efficient_out = flash_attn(x, mask=mask)

