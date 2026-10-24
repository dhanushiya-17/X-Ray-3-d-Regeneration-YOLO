"""
multiview_net.py
================
Multi-view X-ray -> 3D femur occupancy network.

Each view (AP and/or lateral) is encoded by a small 2D U-Net into per-pixel
features. Because the images are aligned with the anatomical grid (see
build_biplanar_dataset.py), AP features are back-projected along Z and lateral
features along Y, giving a 3D feature volume in which every voxel sees the two
rays that pass through it. A 3D U-Net decodes that volume into occupancy
logits. This follows the "2D encoder, expand to 3D, 3D decoder" family that
performed best for femurs in the xrayto3D benchmark (NeurIPS 2023) and
Swin-X2S (2025), kept small enough to train on an 8 GB GPU.

Works with one view ("ap") or two ("ap", "lat"); the single-view model must
infer depth entirely from the learned prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(conv, norm, cin, cout):
    return nn.Sequential(conv(cin, cout, 3, padding=1), norm(cout), nn.LeakyReLU(0.1, inplace=True),
                         conv(cout, cout, 3, padding=1), norm(cout), nn.LeakyReLU(0.1, inplace=True))


class UNet(nn.Module):
    def __init__(self, dims, cin, cout, base, levels=3):
        super().__init__()
        conv = nn.Conv2d if dims == 2 else nn.Conv3d
        norm = (lambda c: nn.GroupNorm(min(8, c), c))
        self.pool = nn.MaxPool2d(2) if dims == 2 else nn.MaxPool3d(2)
        self.mode = "bilinear" if dims == 2 else "trilinear"
        chans = [base * 2 ** i for i in range(levels + 1)]
        self.down = nn.ModuleList([_block(conv, norm, cin if i == 0 else chans[i - 1], chans[i])
                                   for i in range(levels)])
        self.mid = _block(conv, norm, chans[levels - 1], chans[levels])
        self.up = nn.ModuleList([_block(conv, norm, chans[i + 1] + chans[i], chans[i])
                                 for i in reversed(range(levels))])
        self.head = conv(chans[0], cout, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for block, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[2:], mode=self.mode, align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


class MultiViewFemurNet(nn.Module):
    """views: tuple of "ap" and/or "lat". Input images are (B, 1, X, Y) for AP
    and (B, 1, X, Z) for lateral; output logits are (B, 1, X, Y, Z)."""

    def __init__(self, views=("ap", "lat"), feat=8, base2d=16, base3d=16):
        super().__init__()
        self.views = tuple(views)
        self.encoders = nn.ModuleDict({v: UNet(2, 1, feat, base2d) for v in self.views})
        self.decoder = UNet(3, feat * len(self.views), 1, base3d)

    def forward(self, images, depth):
        """images: dict view -> tensor; depth: (Y, Z) size of the output grid."""
        ny, nz = depth
        volumes = []
        if "ap" in self.views:
            f = self.encoders["ap"](images["ap"])            # B, C, X, Y
            volumes.append(f.unsqueeze(-1).expand(-1, -1, -1, -1, nz))
        if "lat" in self.views:
            f = self.encoders["lat"](images["lat"])          # B, C, X, Z
            volumes.append(f.unsqueeze(3).expand(-1, -1, -1, ny, -1))
        return self.decoder(torch.cat(volumes, dim=1))


def dice_bce_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    inter = (prob * target).sum(dims)
    dice = 1.0 - (2.0 * inter + 1.0) / (prob.sum(dims) + target.sum(dims) + 1.0)
    return bce + dice.mean()
