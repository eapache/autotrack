"""Score embeddings and reference-audio features for the SoulX checkpoint."""
import math

import torch
from torch import nn


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ref: https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/modules.py#L247
class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class MelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        # The checkpoint supplies both the mel filterbank and analysis window.
        self.register_buffer("mel_basis", torch.empty(128, 961))
        self.register_buffer("hann_window", torch.empty(1920))

    def forward(self, waveform):
        padded = torch.nn.functional.pad(waveform.unsqueeze(1), (720, 720), mode="reflect")
        spec = torch.stft(padded.squeeze(1), 1920, hop_length=480, window=self.hann_window,
                          center=False, return_complex=True)
        magnitude = (torch.view_as_real(spec).square().sum(-1) + 1e-9).sqrt()
        return (self.mel_basis @ magnitude).clamp_min(1e-5).log()


class MelSpectrogramEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MelSpectrogram()

    def forward(self, waveform):
        mel = self.model(waveform).transpose(1, 2)
        return (mel + 4.92) / math.sqrt(8.14)
