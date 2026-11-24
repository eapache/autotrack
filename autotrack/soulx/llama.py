"""SoulX diffusion decoder, adapted to native PyTorch for score inference."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 1.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LlamaAdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6, dim_cond=1024):
        super().__init__()
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        nn.init.zeros_(self.to_weight.weight)
        nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        self._is_hf_initialized = True  # disable automatic init

    def forward(self, hidden_states, cond_embedding):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.to_weight(cond_embedding)
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)

        return (weight * hidden_states).to(input_dtype)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, x, cos, sin, mask):
        batch, length, width = x.shape

        def project(layer):
            return layer(x).reshape(batch, length, self.heads, -1).transpose(1, 2)

        def rotate(v):
            left, right = v.chunk(2, dim=-1)
            return v * cos + torch.cat((-right, left), dim=-1) * sin

        result = F.scaled_dot_product_attention(
            rotate(project(self.q_proj)), rotate(project(self.k_proj)), project(self.v_proj),
            attn_mask=mask,
        )
        return self.o_proj(result.transpose(1, 2).reshape(batch, length, width))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj = nn.Linear(width, width * 4, bias=False)
        self.up_proj = nn.Linear(width, width * 4, bias=False)
        self.down_proj = nn.Linear(width * 4, width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.self_attn = Attention(width, heads)
        self.mlp = MLP(width)
        self.input_layernorm = LlamaAdaptiveRMSNorm(width, dim_cond=width)
        self.post_attention_layernorm = LlamaAdaptiveRMSNorm(width, dim_cond=width)

    def forward(self, x, step, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x, step), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x, step))


class DiffLlama(nn.Module):
    def __init__(self, mel_dim, hidden_size, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(hidden_size, num_heads) for _ in range(num_layers))
        self.head_dim = hidden_size // num_heads
        self.norm = LlamaAdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mel_mlp = nn.Sequential(
            nn.Linear(mel_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mel_out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, mel_dim),
        )

    def forward(self, x, diffusion_step, cond, x_mask):
        x = self.mel_mlp(x) + self.cond_mlp(cond)
        step = self.diff_step_mlp(self.diff_step_embedding(diffusion_step))
        positions = torch.arange(x.shape[1], dtype=torch.float32, device=x.device)
        dims = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=x.device)
        angles = torch.outer(positions, 1.0 / (10000 ** (dims / self.head_dim)))
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        mask = x_mask[:, None, None, :].bool() if x_mask is not None else None
        for layer in self.layers:
            x = layer(x, step, cos, sin, mask)
        return self.mel_out_mlp(self.norm(x, step))
