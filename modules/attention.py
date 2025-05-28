import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        device = x.device
        
        b, c, h, w = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        
        y = (y_avg + y_max) / 2
        
        self.fc = self.fc.to(device)
        y = self.fc(y).view(b, c, 1, 1)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        
        self.spatial = self.spatial.to(device)
        spatial = self.spatial(spatial)
        
        x = x * y.expand_as(x)
        x = x * spatial
        
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, hsi_channels, sar_channels):
        super(CrossModalAttention, self).__init__()
        self.cross_modal_attention = nn.Sequential(
            nn.Conv2d(hsi_channels + sar_channels, hsi_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(hsi_channels),
            nn.Conv2d(hsi_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, hsi, sar):
        cross_attention = torch.cat([hsi, sar], dim=1)
        attention_map = self.cross_modal_attention(cross_attention)
        return sar * attention_map

class FeatureEnhanceModule(nn.Module):
    def __init__(self, channels):
        super(FeatureEnhanceModule, self).__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.enhance(x)

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(MultiScaleFeatureFusion, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)
        return x_out

class FFusAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFusAttention, self).__init__()
        self.fusion = MultiScaleFeatureFusion(in_channels * 3, out_channels)

    def forward(self, x1, x2, x4):
        return self.fusion(x1, x2, x4)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, is_3d=False, use_se=True,
                 use_residual=True, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        self.use_se = use_se

        if is_3d:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.SiLU(inplace=True)
            )
            if use_se:
                self.se = None
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
            if use_se:
                self.se = SEModule(out_channels)

    def forward(self, x):
        identity = x

        x = self.conv(x)

        if self.use_se:
            if isinstance(self.se, SEModule):
                x = self.se(x)
            else:
                if len(x.shape) == 5:  # [B, C, D, H, W]
                    B, C, D, H, W = x.shape
                    x_reshape = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                    se_module = SEModule(C).to(x_reshape.device)
                    x_reshape = se_module(x_reshape)
                    x = x_reshape.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)

        if self.use_residual:
            x = x + identity

        return x

class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FFusAttention(in_channels * 3, out_channels)

    def forward(self, x1, x2, x4, last=False):
        x_fused = self.fusion_conv(x1, x2, x4)
        return x_fused


if __name__ == '__main__':

    block = MSAA(in_channels=64, out_channels=128)
    x1 = torch.randn(1, 64, 64, 64)
    x2 = torch.randn(1, 64, 64, 64)
    x4 = torch.randn(1, 64, 64, 64)

    output = block(x1, x2, x4)

    # Print the shapes of the inputs and the output
    print(x1.size())
    print(x2.size())
    print(x4.size())
    print(output.size())
