"""Restormer style image restoration model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.num_heads

        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        attention = torch.matmul(q.transpose(-2, -1), k) * self.temperature
        attention = attention.softmax(dim=-1)
        out = torch.matmul(v, attention.transpose(-2, -1))
        out = out.reshape(b, c, h, w)
        return self.project(out)


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor=2.66):
        super().__init__()
        hidden = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GDFN(channels)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Restormer(nn.Module):
    def __init__(self, n_channels_in, n_classes_out, start_filters=40):
        super().__init__()
        sf = start_filters
        channels = [sf, sf * 2, sf * 4, sf * 8, sf * 16]
        heads = [1, 2, 4, 8, 8]

        self.embed = nn.Conv2d(n_channels_in, channels[0], kernel_size=3, padding=1)
        self.encoder1 = nn.Sequential(TransformerBlock(channels[0], heads[0]), TransformerBlock(channels[0], heads[0]))
        self.encoder2 = nn.Sequential(TransformerBlock(channels[1], heads[1]), TransformerBlock(channels[1], heads[1]))
        self.encoder3 = nn.Sequential(TransformerBlock(channels[2], heads[2]), TransformerBlock(channels[2], heads[2]))
        self.encoder4 = nn.Sequential(TransformerBlock(channels[3], heads[3]), TransformerBlock(channels[3], heads[3]))
        self.latent = nn.Sequential(TransformerBlock(channels[4], heads[4]), TransformerBlock(channels[4], heads[4]))

        self.down1 = nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2)
        self.down2 = nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2)
        self.down3 = nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2)
        self.down4 = nn.Conv2d(channels[3], channels[4], kernel_size=2, stride=2)

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)

        self.reduce4 = nn.Conv2d(channels[3] * 2, channels[3], kernel_size=1)
        self.reduce3 = nn.Conv2d(channels[2] * 2, channels[2], kernel_size=1)
        self.reduce2 = nn.Conv2d(channels[1] * 2, channels[1], kernel_size=1)
        self.reduce1 = nn.Conv2d(channels[0] * 2, channels[0], kernel_size=1)

        self.decoder4 = nn.Sequential(TransformerBlock(channels[3], heads[3]), TransformerBlock(channels[3], heads[3]))
        self.decoder3 = nn.Sequential(TransformerBlock(channels[2], heads[2]), TransformerBlock(channels[2], heads[2]))
        self.decoder2 = nn.Sequential(TransformerBlock(channels[1], heads[1]), TransformerBlock(channels[1], heads[1]))
        self.decoder1 = nn.Sequential(TransformerBlock(channels[0], heads[0]), TransformerBlock(channels[0], heads[0]))

        self.outc = nn.Conv2d(channels[0], n_classes_out, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(n_channels_in, n_classes_out, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        residual = self.skip(x)
        e1 = self.encoder1(self.embed(x))
        e2 = self.encoder2(self.down1(e1))
        e3 = self.encoder3(self.down2(e2))
        e4 = self.encoder4(self.down3(e3))
        latent = self.latent(self.down4(e4))

        d4 = self.up4(latent)
        d4 = self.decoder4(self.reduce4(torch.cat([d4, e4], dim=1)))
        d3 = self.up3(d4)
        d3 = self.decoder3(self.reduce3(torch.cat([d3, e3], dim=1)))
        d2 = self.up2(d3)
        d2 = self.decoder2(self.reduce2(torch.cat([d2, e2], dim=1)))
        d1 = self.up1(d2)
        d1 = self.decoder1(self.reduce1(torch.cat([d1, e1], dim=1)))

        output = self.outc(d1) + residual
        return self.final_activation(output)
