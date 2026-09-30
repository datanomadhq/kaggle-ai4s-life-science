"""U-Net for brightfield -> Cell Painting translation with an optional heteroscedastic head.

With ``uncertainty=True`` the network predicts, for every output channel, a mean ``mu`` and a
log-scale ``log_b`` of a Laplace distribution; the training loss is the Laplace negative
log-likelihood |y - mu| / b + log b, so ``b`` is a calibrated estimate of the expected absolute
error and can be shown as a per-pixel "trust map".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 5, base: int = 32, depth: int = 4, uncertainty: bool = True):
        super().__init__()
        self.uncertainty = uncertainty
        self.out_ch = out_ch
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.enc.append(conv_block(c, ch))
            c = ch
        self.bottleneck = conv_block(chs[-2], chs[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.dec.append(conv_block(chs[i - 1] * 2, chs[i - 1]))
        self.head_mu = nn.Conv2d(chs[0], out_ch, 1)
        self.head_logb = nn.Conv2d(chs[0], out_ch, 1) if uncertainty else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, s], dim=1))
        mu = self.head_mu(x)
        logb = self.head_logb(x).clamp(-7, 3) if self.uncertainty else None
        return mu, logb


class LinearPixel(nn.Module):
    """Baseline: a per-pixel affine map of the brightfield intensity (no spatial context)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 5):
        super().__init__()
        self.lin = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.lin(x), None


class SmallCNN(nn.Module):
    """Baseline: shallow fully-convolutional net with a 25x25-pixel receptive field, no down-sampling."""

    def __init__(self, in_ch: int = 1, out_ch: int = 5, width: int = 32, layers: int = 6):
        super().__init__()
        mods, c = [], in_ch
        for _ in range(layers):
            mods += [nn.Conv2d(c, width, 5, padding=2), nn.ReLU(inplace=True)]
            c = width
        mods.append(nn.Conv2d(c, out_ch, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x), None


def build_model(name: str, **kw) -> nn.Module:
    if name == "unet":
        return UNet(**kw)
    if name == "unet_nounc":
        kw = dict(kw)
        kw["uncertainty"] = False
        return UNet(**kw)
    if name == "linear":
        return LinearPixel(in_ch=kw.get("in_ch", 1), out_ch=kw.get("out_ch", 5))
    if name == "smallcnn":
        return SmallCNN(in_ch=kw.get("in_ch", 1), out_ch=kw.get("out_ch", 5))
    raise ValueError(name)


# ----------------------------------------------------------------------------- losses

def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    g = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM per (sample, channel); x, y: [B, C, H, W]."""
    C = x.shape[1]
    w = _gaussian_window(window_size, device=x.device).repeat(C, 1, 1, 1)
    pad = window_size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=C)
    mu_y = F.conv2d(y, w, padding=pad, groups=C)
    sxx = F.conv2d(x * x, w, padding=pad, groups=C) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=C) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=C) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(2, 3))


def translation_loss(mu: torch.Tensor, logb: torch.Tensor | None, y: torch.Tensor,
                     ssim_weight: float = 0.0, nll: bool | str = "detached") -> tuple[torch.Tensor, dict]:
    """L1 / Laplace-NLL translation loss.

    nll="detached" (default): the mean is trained with plain L1 and the scale head with the Laplace NLL of the
    *detached* residuals, so the uncertainty head can never trade accuracy for a large predicted scale (the
    pitfall of joint heteroscedastic training, Seitzer et al. 2022). nll="joint": classic joint NLL
    |y-mu|/b + log b (Kendall & Gal 2017). nll=False/"none": plain L1 (scale head, if any, untrained).
    """
    if nll is True:
        nll = "joint"
    err = (mu - y).abs()
    if logb is not None and nll == "joint":
        b = torch.exp(logb)
        base = (err / b + logb).mean()
    elif logb is not None and nll == "detached":
        b = torch.exp(logb)
        base = err.mean() + (err.detach() / b + logb).mean()
    else:
        base = err.mean()
    parts = {"l1": err.mean().item(), "base": base.item()}
    loss = base
    if ssim_weight > 0:
        s = ssim(mu.clamp(0, 1), y.clamp(0, 1)).mean()
        loss = loss + ssim_weight * (1 - s)
        parts["ssim"] = s.item()
    return loss, parts
