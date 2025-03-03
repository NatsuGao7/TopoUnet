
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DownWithPI(nn.Sequential):
    def __init__(self, in_channels, out_channels, pi_channels, bilinear=True):
        # 这里添加 pi_channels 参数
        super(DownWithPI, self).__init__()
        self.pi_channels = pi_channels
        self.bilinear = bilinear
        
        # 使用 MaxPool2d 进行下采样，然后执行 DoubleConv
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv = DoubleConv(in_channels + pi_channels, out_channels)  # 在这里处理 PI

    def forward(self, x: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        # 调整 PI 的尺寸
        pi = F.interpolate(pi, size=x.shape[2:], mode='bilinear', align_corners=True)  # 调整 PI 的尺寸与输入图像一致
        x = torch.cat((x, pi), dim=1)  # 将 PI 和图像拼接

        # 下采样和卷积操作
        x = self.pool(x)  # 下采样
        x = self.conv(x)  # 卷积操作
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, pi_channels=4):
        super(Up, self).__init__()
        self.bilinear = bilinear
        self.pi_channels = pi_channels

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + pi_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + pi_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # 计算 padding 以匹配 x2 的尺寸
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # 调整 PI 尺寸与当前层一致
        if self.pi_channels > 0:
            pi = F.interpolate(pi, size=x1.size()[2:], mode='bilinear', align_corners=True)
            x1 = torch.cat([x1, pi], dim=1)

        # 拼接 decoder 输出和 encoder 输出
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class UNetWithPI(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 pi_channels: int = 4,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNetWithPI, self).__init__()
        self.in_channels = in_channels
        self.pi_channels = pi_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 输入卷积层，接受原始图像和持久性图
        self.in_conv = DoubleConv(in_channels + pi_channels, base_c)

        # 定义 U-Net 的其余部分，使用 DownWithPI 替代 Down
        self.down1 = DownWithPI(base_c, base_c * 2, pi_channels)
        self.down2 = DownWithPI(base_c * 2, base_c * 4, pi_channels)
        self.down3 = DownWithPI(base_c * 4, base_c * 8, pi_channels)
        factor = 2 if bilinear else 1
        self.down4 = DownWithPI(base_c * 8, base_c * 16 // factor, pi_channels)

        # 在 decoder 中插入拓扑特征
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear, pi_channels)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear, pi_channels)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear, pi_channels)
        self.up4 = Up(base_c * 2, base_c, bilinear, pi_channels)

        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor, pi: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 在前向传播中，将 PI 数据仅插入 encoder 部分
        pi = F.interpolate(pi, size=x.shape[2:], mode='bilinear', align_corners=True)  # 调整 PI 的尺寸与输入图像一致
        x = torch.cat((x, pi), dim=1)  # 在输入阶段拼接 PI

        # 通过 U-Net 的 encoder，插入 PI
        x1 = self.in_conv(x)
        x2 = self.down1(x1, pi)
        x3 = self.down2(x2, pi)
        x4 = self.down3(x3, pi)
        x5 = self.down4(x4, pi)

        # 通过 U-Net 的 decoder，插入 PI
        x = self.up1(x5, x4, pi)
        x = self.up2(x, x3, pi)
        x = self.up3(x, x2, pi)
        x = self.up4(x, x1, pi)

        logits = self.out_conv(x)

        return {"out": logits}

