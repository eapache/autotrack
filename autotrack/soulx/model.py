"""Score-conditioned SoulX-Singer inference with the released checkpoint."""
import torch
from torch import nn

from .diffusion import CFMDecoder
from .encoder import ConvNeXtV2Block, MelSpectrogramEncoder
from .vocoder import Vocoder


class SoulXSinger(nn.Module):
    def __init__(self):
        super().__init__()
        self.note_text_encoder = nn.Embedding(3000, 512)
        self.note_pitch_encoder = nn.Embedding(256, 512)
        self.note_type_encoder = nn.Embedding(256, 512)
        self.f0_encoder = nn.Embedding(361, 512)
        self.preflow = nn.Sequential(*(ConvNeXtV2Block(512, 1024) for _ in range(4)))
        self.cfm_decoder = CFMDecoder()
        self.mel = MelSpectrogramEncoder()
        self.vocoder = Vocoder()

    def forward(self, prompt, target, steps):
        def join(key):
            return torch.cat([prompt[key], target[key]], dim=1)

        notes = (self.note_pitch_encoder(join('note_pitch')) + self.note_type_encoder(join('note_type'))
                 + self.note_text_encoder(join('phoneme')))
        frame_indices = torch.cat([prompt['mel2note'], target['mel2note'] + prompt['phoneme'].shape[1]], dim=1)
        features = self.preflow(notes).gather(1, frame_indices.unsqueeze(-1).expand(-1, -1, notes.shape[-1]))
        # In score mode the MIDI notes supply pitch. F0 bin zero marks an absent contour.
        features = features + self.f0_encoder(torch.zeros_like(frame_indices))
        mel = self.cfm_decoder(self.mel(prompt['waveform']), features, steps)
        return self.vocoder(mel.transpose(1, 2))


def load_model(checkpoint_path, device):
    model = SoulXSinger()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval().to(device)
