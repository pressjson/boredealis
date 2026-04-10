"""Loss helpers for smoother training."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19

if not os.path.exists("local_settings.py"):
    import settings
else:
    import local_settings as settings


def warp_frame(frame, flow):
    batch_size, _, height, width = frame.size()
    xx = torch.arange(0, width).view(1, -1).repeat(height, 1)
    yy = torch.arange(0, height).view(-1, 1).repeat(1, width)
    xx = xx.view(1, 1, height, width).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, height, width).repeat(batch_size, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(frame.device)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(width - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(height - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(frame, vgrid, align_corners=True, padding_mode="border")


class LAMBDAS:
    def __init__(self, l1, rec, l1_perc, rec_perc):
        self.l1 = l1
        self.rec = rec
        self.l1_perc = l1_perc
        self.rec_perc = rec_perc


class LOSSES:
    def __init__(self, total, temp, rec, temp_perc, rec_perc):
        self.total_loss = total
        self.temp_loss = temp
        self.rec_loss = rec
        self.temp_perc_loss = temp_perc
        self.rec_perc_loss = rec_perc


class DeflickerLoss(nn.Module):
    def __init__(self, lambda_values, device=""):
        super().__init__()
        self.LAMBDAS = lambda_values
        self.l1_loss = nn.L1Loss()
        vgg = vgg19(weights=VGG19_Weights).features
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vgg = vgg[:12].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, output_t, input_t, output_prev, flow, occlusion_mask=None):
        if input_t.device != output_t.device:
            input_t = input_t.to(output_t.device)
        rec_loss = self.l1_loss(output_t, input_t)
        vgg_device = self.mean.device
        if occlusion_mask is not None and occlusion_mask.device != output_t.device:
            occlusion_mask = occlusion_mask.to(output_t.device)
        warped_prev = warp_frame(output_prev, flow)

        out_norm = (output_t.to(vgg_device) - self.mean) / self.std
        in_norm = (input_t.to(vgg_device) - self.mean) / self.std
        warped_norm = (warped_prev.to(vgg_device) - self.mean) / self.std
        out_feat = self.vgg(out_norm)
        in_feat = self.vgg(in_norm)
        rec_perc_loss = self.l1_loss(out_feat, in_feat)
        with torch.no_grad():
            warped_feat = self.vgg(warped_norm)

        if occlusion_mask is not None:
            temp_diff = torch.abs(output_t - warped_prev) * occlusion_mask
            temp_loss = torch.mean(temp_diff)
            mask_vgg = occlusion_mask.to(vgg_device)
            mask_feat = F.interpolate(mask_vgg, size=out_feat.shape[-2:], mode="nearest")
            perc_diff = torch.abs(out_feat - warped_feat) * mask_feat
            temp_perc_loss = torch.mean(perc_diff)
        else:
            temp_loss = self.l1_loss(output_t, warped_prev)
            temp_perc_loss = self.l1_loss(out_feat, warped_feat)

        total_loss = (
            self.LAMBDAS.l1 * temp_loss
            + self.LAMBDAS.rec * rec_loss
            + self.LAMBDAS.l1_perc * temp_perc_loss.to(output_t.device)
            + self.LAMBDAS.rec_perc * rec_perc_loss.to(output_t.device)
        )
        return LOSSES(total_loss, temp_loss, rec_loss, temp_perc_loss, rec_perc_loss)


def generate_circle_mask(height=608, width=608, radius=250, vertical_offset=15, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    center_x = width // 2
    center_y = (height // 2) - vertical_offset
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    mask = (dist_sq <= radius**2).float()
    return mask.view(1, 1, height, width)


def resolve_vgg_device(vgg_device_arg, train_device):
    if vgg_device_arg is None:
        if settings.USE_VGG_DEVICE:
            return torch.device(f"cuda:{settings.VGG_DEVICE_ID}")
        return train_device

    requested_device = vgg_device_arg.strip().lower()
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        return torch.device("cuda")
    if requested_device.startswith("cuda:"):
        return torch.device(requested_device)
    return torch.device(f"cuda:{int(requested_device)}")
