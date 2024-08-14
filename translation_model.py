import torch
import torch.nn as nn


class translation_ray(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.TransformerEncoder(nheads=512,)