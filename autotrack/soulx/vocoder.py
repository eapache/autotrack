"""Vocos waveform decoder for the released SoulX model."""
import torch
from torch import nn


class ISTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = self.win_length = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, spec):
        T = spec.shape[-1]
        pad = (self.win_length - self.hop_length) // 2
        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(nn.Module):
    def __init__(self, dim, n_fft, hop_length):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft, hop_length)

    def forward(self, x):
        magnitude, phase = self.out(x).transpose(1, 2).chunk(2, dim=1)
        magnitude = magnitude.exp().clamp_max(100)
        return self.istft(magnitude * (phase.cos() + 1j * phase.sin()))


class ConvNeXtBlock(nn.Module):
    """One-dimensional ConvNeXt block, adapted from facebookresearch/ConvNeXt."""
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        hidden = self.norm(self.dwconv(x).transpose(1, 2))
        hidden = self.pwconv2(self.act(self.pwconv1(hidden)))
        return x + (self.gamma * hidden).transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv1d(128, 1024, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(1024, eps=1e-6)
        self.convnext = nn.Sequential(*(ConvNeXtBlock(1024, 4096) for _ in range(30)))
        self.final_layer_norm = nn.LayerNorm(1024, eps=1e-6)

    def forward(self, x):
        hidden = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        return self.final_layer_norm(self.convnext(hidden).transpose(1, 2))


class Vocos(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = VocosBackbone()
        self.head = ISTFTHead(1024, 1920, 480)

    def forward(self, mel):
        return self.head(self.backbone(mel))[:, None, :]


class Vocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Vocos()

    def forward(self, mel):
        return self.model(mel)
