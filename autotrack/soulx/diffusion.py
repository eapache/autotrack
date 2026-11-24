# https://github.com/open-mmlab/Amphion/blob/main/models/svc/flow_matching_transformer/fmt_model.py

# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from .llama import DiffLlama


class FlowMatchingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_emb = nn.Linear(512, 1024)
        self.diff_estimator = DiffLlama(mel_dim=128, hidden_size=1024, num_layers=22, num_heads=16)

    def forward(self, cond, prompt, steps):
        h = 1.0 / steps
        prompt_len = prompt.shape[1]
        target_len = cond.shape[1] - prompt_len
        x_mask = cond.new_ones(cond.shape[0], target_len)
        xt_mask = cond.new_ones(cond.shape[:2])
        xt = torch.randn((cond.shape[0], target_len, 128), dtype=cond.dtype, device=cond.device)
        unconditioned = torch.zeros_like(cond[:, :target_len])
        for i in range(steps):
            t = xt.new_full((xt.shape[0],), (i + 0.5) * h)
            flow = self.diff_estimator(torch.cat([prompt, xt], dim=1), t, cond, xt_mask)
            flow = flow[:, prompt_len:, :]
            uncond_flow = self.diff_estimator(xt, t, unconditioned, x_mask)
            guided = flow + 3 * (flow - uncond_flow)
            rescaled = guided * flow.std() / guided.std()
            xt = xt + (0.75 * rescaled + 0.25 * guided) * h
        return xt


class CFMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FlowMatchingTransformer()

    def forward(self, prompt_mel, features, steps):
        return self.model(self.model.cond_emb(features), prompt_mel, steps)
