import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

# ─────────────────── Attention 계열 ───────────────────
class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 3):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # Spatial Attention
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel
        avg = self.mlp(self.avg_pool(x))
        maxv = self.mlp(x.amax(dim=(-2, -1), keepdim=True))
        x = x * self.sigmoid(avg + maxv)
        # Spatial
        avg_map = x.mean(1, keepdim=True)
        max_map, _ = x.max(1, keepdim=True)
        x = x * self.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], 1)))
        return x


class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)     # B,C,1 → B,1,C
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)        # B,C,1,1
        return x * self.sigmoid(y)


class CBAM_ECA(nn.Module):
    """CBAM 뒤에 ECA를 연달아 붙인 모듈"""
    def __init__(self, channels, reduction=8, spatial_kernel=3, eca_kernel=3):
        super().__init__()
        self.cbam = CBAM(channels, reduction, spatial_kernel)
        self.eca = ECA(channels, eca_kernel)

    def forward(self, x):
        return self.eca(self.cbam(x))


# ─────────────────── SE 모듈 ───────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


# ─────────────────── ASPP ───────────────────
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 3, 6, 9)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=r, dilation=r, bias=False)
            for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(len(rates)*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = [F.relu(b(x), inplace=True) for b in self.blocks]
        return self.project(torch.cat(feats, 1))


# ─────────────────── Residual Decoder Block + SE ───────────────────
class ResidualDecoderBlock(nn.Module):
    """
    업샘플 → Concat(skip) → 3×3 DWConv + PWConv ×2 → SE → Residual Add
    depthwise=True이면 Depthwise-Separable Conv 사용 (파라미터 감소)
    """
    def __init__(self, in_channels, skip_channels, out_channels,
                 depthwise=True, se_reduction=8):
        super().__init__()
        total_in = in_channels + skip_channels

        # identity 경로
        self.proj = (nn.Identity() if total_in == out_channels
                     else nn.Conv2d(total_in, out_channels, 1, bias=False))

        def DWConv(ic, oc):
            if not depthwise:
                return nn.Conv2d(ic, oc, 3, padding=1, bias=False)

            return nn.Sequential(
                nn.Conv2d(ic, ic, 3, padding=1, groups=ic, bias=False),
                nn.Conv2d(ic, oc, 1, bias=False)
            )

        self.conv1 = DWConv(total_in, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = DWConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction=se_reduction)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], 1)

        identity = self.proj(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                # SE 통과
        out = self.relu2(out + identity)  # Residual
        return out


# ─────────────────── Main Network with Deep Supervision ───────────────────
class submission_20214173(nn.Module):
    """
    MobileNetV2 encoder + CBAM_ECA + Dual ASPP + Residual Decoder(SE) + Deep Supervision
    """
    def __init__(self, in_channels=3, num_classes=21,
                 mid_channels=128, pretrained=False,
                 deep_sup_weight=0.4):
        super().__init__()
        self.deep_sup_weight = deep_sup_weight

        backbone = mobilenet_v2(pretrained=False)
        if in_channels != 3:
            backbone.features[0][0] = nn.Conv2d(in_channels, 32, 3,
                                                 stride=2, padding=1, bias=False)

        # Encoder
        self.enc1 = backbone.features[:4]   # stride 2
        self.enc2 = backbone.features[4:7]  # stride 4
        self.enc3 = backbone.features[7:14] # stride 8

        # 채널 수 계산
        c1 = self._get_dim(self.enc1, in_channels)
        c2 = self._get_dim(self.enc2, c1)
        c3 = self._get_dim(self.enc3, c2)

        # Attention
        self.att1 = CBAM_ECA(c1)
        self.att2 = CBAM_ECA(c2)
        self.att3 = CBAM_ECA(c3)

        # ASPP
        self.aspp2 = ASPP(c2, mid_channels)
        self.aspp3 = ASPP(c3, mid_channels)

        # Decoder (Residual + SE)
        self.dec2 = ResidualDecoderBlock(mid_channels, mid_channels, mid_channels)
        self.dec1 = ResidualDecoderBlock(mid_channels, c1, mid_channels // 2)

        # Classifiers
        self.cls_main = nn.Conv2d(mid_channels // 2, num_classes, 1)
        self.cls_aux  = nn.Conv2d(mid_channels,        num_classes, 1)  # deep sup

    def _get_dim(self, module, in_ch):
        with torch.no_grad():
            return module(torch.zeros(1, in_ch, 256, 256)).shape[1]

    def forward(self, x):
        h, w = x.shape[2:]
        # Encoder
        f1 = self.att1(self.enc1(x))
        f2 = self.att2(self.enc2(f1))
        f3 = self.att3(self.enc3(f2))

        # ASPP
        a2 = self.aspp2(f2)
        a3 = self.aspp3(f3)

        # Decoder
        d2 = self.dec2(a3, a2)  # stride 4
        d1 = self.dec1(d2, f1)  # stride 2

        # logits
        main = self.cls_main(d1)
        main = F.interpolate(main, size=(h, w), mode='bilinear', align_corners=False)

       
        return main


