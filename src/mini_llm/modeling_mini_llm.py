import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_mini_llm import MiniLlmConfig
except ImportError:
    from configuration_mini_llm import MiniLlmConfig


class MiniLlmRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * n).type_as(x) * self.w


def mini_rope_cache(T, hd, device, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def mini_apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[: x.size(2)], sin[: x.size(2)]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class MiniLlmAttention(nn.Module):
    def __init__(self, config: MiniLlmConfig):
        super().__init__()
        d = config.hidden_size
        self.h = config.num_attention_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        q, k = mini_apply_rope(q, cos, sin), mini_apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, C))


class MiniLlmSwiGLU(nn.Module):
    def __init__(self, config: MiniLlmConfig):
        super().__init__()
        d, ffn = config.hidden_size, config.intermediate_size
        self.w1 = nn.Linear(d, ffn, bias=False)
        self.w3 = nn.Linear(d, ffn, bias=False)
        self.w2 = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MiniLlmBlock(nn.Module):
    def __init__(self, config: MiniLlmConfig):
        super().__init__()
        self.n1, self.n2 = MiniLlmRMSNorm(config.hidden_size, config.rms_norm_eps), MiniLlmRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.attn, self.mlp = MiniLlmAttention(config), MiniLlmSwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class MiniLlmForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniLlmConfig
    base_model_prefix = "mini_llm"
    supports_gradient_checkpointing = False
    _tied_weights_keys = {"head.weight": "emb.weight"}

    def __init__(self, config: MiniLlmConfig):
        super().__init__(config)
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(MiniLlmBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = MiniLlmRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.head.weight = self.emb.weight
        cos, sin = mini_rope_cache(
            config.max_position_embeddings,
            config.hidden_size // config.num_attention_heads,
            "cpu",
            config.rope_theta,
        )
        self.register_buffer("cos", cos, persistent=True)
        self.register_buffer("sin", sin, persistent=True)
        self.post_init()
        for n, p in self.named_parameters():
            if n.endswith("attn.o.weight") or n.endswith("mlp.w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * config.num_hidden_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, value):
        self.emb = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        kwargs.pop("past_key_values", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("position_ids", None)
        kwargs.pop("inputs_embeds", None)
        if input_ids is None:
            raise ValueError("input_ids is required")
        x = self.emb(input_ids)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


try:
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("mini_llm", MiniLlmConfig)
    AutoModelForCausalLM.register(MiniLlmConfig, MiniLlmForCausalLM)
except Exception:
    pass
