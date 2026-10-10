import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * n).type_as(x) * self.w


def rope_cache(T, hd, device):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[: x.size(2)], sin[: x.size(2)]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c.heads
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.o = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.dim, c.ffn, bias=False)
        self.w3 = nn.Linear(c.dim, c.ffn, bias=False)
        self.w2 = nn.Linear(c.ffn, c.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.dim), RMSNorm(c.dim)
        self.attn, self.mlp = Attention(c), SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class LLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = RMSNorm(c.dim)
        self.head = nn.Linear(c.dim, c.vocab, bias=False)
        self.head.weight = self.emb.weight
        cos, sin = rope_cache(c.ctx, c.dim // c.heads, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("attn.o.weight") or n.endswith("mlp.w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n=200, temp=0.8, top_k=50):
        for _ in range(n):
            logits, _ = self(idx[:, -self.c.ctx:])
            logits = logits[:, -1] / temp
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx