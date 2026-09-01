"""Model zoo: the RGB U-Net baseline and its physics-prior variants.

- ``UNet``               - vanilla encoder/decoder U-Net.
- ``PHYSHADENet``        - same topology, ingests RGB + pseudo-shadow prior (4ch).
- ``AttentivePHYSHADENet`` - PHYSHADENet with attention gates on the skips.

All three map ``(B, in_channels, H, W)`` logits to ``(B, out_channels, H, W)``.
"""

import torch
import torch.nn as nn


class ConvStep(nn.Module):
    """Two 3x3 conv + ReLU layers at a fixed feature width."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Vanilla U-Net (RGB in, single-channel mask logits out by default)."""

    def __init__(self, in_channels=3, out_channels=1):

        super().__init__()

        # Downsampling
        self.down1 = ConvStep(in_channels, 64)
        self.down2 = ConvStep(64, 128)
        self.down3 = ConvStep(128, 256)
        self.down4 = ConvStep(256, 512)

        # Bottleneck
        self.bottleneck = ConvStep(512, 1024)

        # Upsampling
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = ConvStep(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ConvStep(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvStep(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ConvStep(128, 64)

        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool(d1)

        d2 = self.down2(p1)
        p2 = self.pool(d2)

        d3 = self.down3(p2)
        p3 = self.pool(d3)

        d4 = self.down4(p3)
        p4 = self.pool(d4)

        bottleneck = self.bottleneck(p4)

        up4 = self.up4(bottleneck)
        merge4 = torch.cat([up4, d4], dim=1)
        c4 = self.conv4(merge4)

        up3 = self.up3(c4)
        merge3 = torch.cat([up3, d3], dim=1)
        c3 = self.conv3(merge3)

        up2 = self.up2(c3)
        merge2 = torch.cat([up2, d2], dim=1)
        c2 = self.conv2(merge2)

        up1 = self.up1(c2)
        merge1 = torch.cat([up1, d1], dim=1)
        c1 = self.conv1(merge1)

        output = self.last_conv(c1)

        return output


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, *args):
        g, x = args
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentivePHYSHADENet(nn.Module):
    """PHYSHADENet with attention gates on every skip connection."""

    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        self.down1 = ConvStep(in_channels, 64)
        self.down2 = ConvStep(64, 128)
        self.down3 = ConvStep(128, 256)
        self.down4 = ConvStep(256, 512)
        self.bottleneck = ConvStep(512, 1024)

        # Attention Gates
        self.att4 = AttentionGate(512, 512, 256)
        self.att3 = AttentionGate(256, 256, 128)
        self.att2 = AttentionGate(128, 128, 64)
        self.att1 = AttentionGate(64, 64, 32)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = ConvStep(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ConvStep(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvStep(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ConvStep(128, 64)

        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool(d1)

        d2 = self.down2(p1)
        p2 = self.pool(d2)

        d3 = self.down3(p2)
        p3 = self.pool(d3)

        d4 = self.down4(p3)
        p4 = self.pool(d4)

        bn = self.bottleneck(p4)

        up4 = self.up4(bn)
        att4 = self.att4(up4, d4)
        merge4 = torch.cat([up4, att4], dim=1)
        c4 = self.conv4(merge4)

        up3 = self.up3(c4)
        att3 = self.att3(up3, d3)
        merge3 = torch.cat([up3, att3], dim=1)
        c3 = self.conv3(merge3)

        up2 = self.up2(c3)
        att2 = self.att2(up2, d2)
        merge2 = torch.cat([up2, att2], dim=1)
        c2 = self.conv2(merge2)

        up1 = self.up1(c2)
        att1 = self.att1(up1, d1)
        merge1 = torch.cat([up1, att1], dim=1)
        c1 = self.conv1(merge1)

        out = self.last_conv(c1)
        return out


class PHYSHADENet(nn.Module):
    """U-Net that ingests RGB + the pseudo-shadow prior as a 4th channel."""

    def __init__(self, in_channels=4, out_channels=1):

        super().__init__()

        # Downsampling
        self.down1 = ConvStep(in_channels, 64)
        self.down2 = ConvStep(64, 128)
        self.down3 = ConvStep(128, 256)
        self.down4 = ConvStep(256, 512)

        # Bottleneck
        self.bottleneck = ConvStep(512, 1024)

        # Upsampling
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = ConvStep(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ConvStep(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvStep(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ConvStep(128, 64)

        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool(d1)

        d2 = self.down2(p1)
        p2 = self.pool(d2)

        d3 = self.down3(p2)
        p3 = self.pool(d3)

        d4 = self.down4(p3)
        p4 = self.pool(d4)

        bottleneck = self.bottleneck(p4)

        up4 = self.up4(bottleneck)
        merge4 = torch.cat([up4, d4], dim=1)
        c4 = self.conv4(merge4)

        up3 = self.up3(c4)
        merge3 = torch.cat([up3, d3], dim=1)
        c3 = self.conv3(merge3)

        up2 = self.up2(c3)
        merge2 = torch.cat([up2, d2], dim=1)
        c2 = self.conv2(merge2)

        up1 = self.up1(c2)
        merge1 = torch.cat([up1, d1], dim=1)
        c1 = self.conv1(merge1)

        output = self.last_conv(c1)

        return output


if __name__ == "__main__":
    for cls, ch in [(UNet, 3), (PHYSHADENet, 4), (AttentivePHYSHADENet, 4)]:
        net = cls(in_channels=ch, out_channels=1).eval()
        out = net(torch.randn(1, ch, 256, 256))
        print(f"{cls.__name__}: (1,{ch},256,256) -> {tuple(out.shape)}")

    print(torch.cuda.is_available())
