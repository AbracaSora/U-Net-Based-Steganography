import torch
from torch import nn


class HidingNet(nn.Module):
    def __init__(self):
        super(HidingNet, self).__init__()

        # 下采样阶段 (Contracting Path)
        self.down1 = self.conv_block(6, 64)       # 输入6通道：cover(3)+secret(3)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)
        self.down5 = self.conv_block(512, 512)
        self.down6 = self.conv_block(512, 512)
        self.down7 = self.conv_block(512, 512)

        # 上采样阶段 (Expanding Path)
        self.up1 = self.up_block(512, 512)
        self.up2 = self.up_block(1024, 512)
        self.up3 = self.up_block(1024, 512)
        self.up4 = self.up_block(1024, 256)
        self.up5 = self.up_block(512, 128)
        self.up6 = self.up_block(256, 64)
        self.up7 = self.up_block(128, 64)

        # 输出层
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.final_activation = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 下采样路径
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # 上采样路径，跳跃连接
        u1 = self.up1(d7)
        u2 = self.up2(torch.cat([u1, d6], dim=1))
        u3 = self.up3(torch.cat([u2, d5], dim=1))
        u4 = self.up4(torch.cat([u3, d4], dim=1))
        u5 = self.up5(torch.cat([u4, d3], dim=1))
        u6 = self.up6(torch.cat([u5, d2], dim=1))
        u7 = self.up7(torch.cat([u6, d1], dim=1))

        out = self.final_conv(u7)
        out = self.final_activation(out)
        return out