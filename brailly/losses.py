from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_heatmap_loss(logit: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0):
    """CenterNet penalty-reduced focal loss."""
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = gt.ge(0.999).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - gt) ** beta * neg
    n = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n


def brailly_loss(out: torch.Tensor, heat, dots, dmask, reg, rmask, w_dots=1.0, w_off=1.0, w_size=0.5):
    l_heat = focal_heatmap_loss(out[:, 0], heat)
    dm = dmask.unsqueeze(1)
    l_dots = (F.binary_cross_entropy_with_logits(out[:, 1:7], dots, reduction="none") * dm).sum() / (dm.sum() * 6).clamp(min=1)
    rm = rmask.unsqueeze(1)
    n = rm.sum().clamp(min=1)
    l_off = (F.l1_loss(out[:, 7:9], reg[:, 0:2], reduction="none") * rm).sum() / (2 * n)
    l_size = (F.l1_loss(out[:, 9:11], reg[:, 2:4], reduction="none") * rm).sum() / (2 * n)
    total = l_heat + w_dots * l_dots + w_off * l_off + w_size * l_size
    return total, {"heat": l_heat.item(), "dots": l_dots.item(), "off": l_off.item(), "size": l_size.item()}
