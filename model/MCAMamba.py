import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from modules.Mamba import MSBlock, FSSBlock, SpecMambaBlock
from modules.attention import ChannelAttentionModule, SpatialAttentionModule, ConvBlock, CrossModalAttention, FeatureEnhanceModule


class SEModule(nn.Module):
    """Squeeze and Excitation Module with Spatial Attention"""

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Add max pooling for better feature capture

        # Enhanced channel attention with two pooling methods
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),  # Use SiLU instead of ReLU for better gradient flow
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        # Small spatial attention component
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Move everything to the input's device
        device = x.device

        # Channel attention
        b, c, h, w = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)

        # Combine two pooling results
        y = (y_avg + y_max) / 2

        # Move fc to device and apply
        self.fc = self.fc.to(device)
        y = self.fc(y).view(b, c, 1, 1)

        # Simple spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)

        # Move spatial attention module to device and apply
        self.spatial = self.spatial.to(device)
        spatial = self.spatial(spatial)

        # Combine channel and spatial attention
        x = x * y.expand_as(x)  # Apply channel attention
        x = x * spatial  # Apply spatial attention

        return x


class ConvBlock(nn.Module):
    """Enhanced convolutional block with SE module and optional residual connection"""

    def __init__(self, in_channels, out_channels, kernel_size, is_3d=False, use_se=True,
                 use_residual=True, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        self.use_se = use_se

        if is_3d:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.SiLU(inplace=True)  # Use SiLU (Swish) instead of ReLU
            )
            if use_se:
                self.se = None  # Specially handled in forward
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)  # Use SiLU (Swish) instead of ReLU
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
                # For 3D data, we need to transform dimensions to apply SE
                if len(x.shape) == 5:  # [B, C, D, H, W]
                    B, C, D, H, W = x.shape
                    x_reshape = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                    # Create SEModule on the same device as input
                    se_module = SEModule(C).to(x_reshape.device)
                    x_reshape = se_module(x_reshape)
                    x = x_reshape.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)

        if self.use_residual:
            x = x + identity

        return x


class Syn_layer(nn.Module):
    def __init__(self, hc1, hc2, sc1, sc2, hsi_N, img_size, d_state, lay_n, expand, channel_reduction=4,
                 spatial_kernel_size=7, dropout_rate=0.0):
        super().__init__()
        N = 9 - lay_n * 2
        N1 = hsi_N - N + 1

        # Feature extraction with attention
        self.hsi_conv = ConvBlock(hc1, hc2, kernel_size=(N, 3, 3), is_3d=True,
                                  use_se=True, padding=0)
        self.sar_conv = ConvBlock(sc1, sc2, kernel_size=3, is_3d=False,
                                  use_se=True, padding=0)

        # Multi-scale Spatial-Channel Attention (MSC-Attn-P) components
        self.MBh = MSBlock(hidden_dim=hc2 * N1, d_state=d_state, expand=expand)
        self.SpBh = SpecMambaBlock(hidden_dim=(img_size - 2) ** 2, d_state=d_state, expand=expand)
        self.MBx = MSBlock(hidden_dim=sc2, d_state=d_state, expand=expand)
        
        # Feature fusion with SSM attention (FExt-Attention)
        self.FSSBlock = FSSBlock(hidden_dim1=hc2 * (hsi_N - N + 1), hidden_dim2=sc2, d_state=d_state, expand=expand)

        # Adaptive attention parameters based on layer depth
        layer_factor = max(1, lay_n + 1)
        adaptive_reduction = max(2, int(channel_reduction / layer_factor))
        adaptive_kernel = min(9, spatial_kernel_size + 2 * lay_n)

        # Channel and Spatial Attention modules for HSI
        self.hsi_channel_attention = ChannelAttentionModule(hc2 * N1, reduction=adaptive_reduction)
        self.hsi_spatial_attention = SpatialAttentionModule(kernel_size=adaptive_kernel)
        
        # Channel and Spatial Attention modules for SAR
        self.sar_channel_attention = ChannelAttentionModule(sc2, reduction=adaptive_reduction)
        self.sar_spatial_attention = SpatialAttentionModule(kernel_size=adaptive_kernel)

        # Cross-modal attention (Cross-Attn)
        self.cross_modal_attention = CrossModalAttention(hc2 * N1, sc2)

        # Feature enhancement modules
        self.hsi_enhance = FeatureEnhanceModule(hc2 * N1)
        self.sar_enhance = FeatureEnhanceModule(sc2)

        # Store layer index for conditional processing
        self.layer_idx = lay_n
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, hsi, sar):
        # Feature extraction
        hsi = self.hsi_conv(hsi)
        sar = self.sar_conv(sar)

        # Save original inputs for residual connections
        hsi_identity = hsi
        sar_identity = sar

        # HSI attention processing
        B, C, N, H, W = hsi.size()
        
        # Spatial attention for HSI
        hsi_spatial = hsi.permute(0, 2, 1, 3, 4).reshape(B * N, C, H, W)
        hsi_spatial = hsi_spatial * self.hsi_spatial_attention(hsi_spatial)
        hsi_spatial = hsi_spatial.reshape(B, N, C, H, W).permute(0, 2, 1, 3, 4)

        # Channel attention for HSI
        hsi_channel = hsi.reshape(B, C * N, H, W)
        hsi_channel = hsi_channel * self.hsi_channel_attention(hsi_channel)
        hsi_channel = hsi_channel.reshape(B, C, N, H, W)

        # SAR attention processing
        sar = sar * self.sar_channel_attention(sar)
        sar = sar * self.sar_spatial_attention(sar)

        # Reshape HSI for cross-modal attention
        hsi_reshape = hsi_channel.reshape(B, C * N, H, W)

        # Apply feature enhancement
        hsi_reshape = hsi_reshape + self.hsi_enhance(hsi_reshape)
        sar = sar + self.sar_enhance(sar)

        # Cross-modal attention (HSI modulates SAR)
        sar = self.cross_modal_attention(hsi_reshape, sar)

        # Feature fusion with residual connections
        hsi = hsi_spatial * 0.4 + hsi_channel * 0.6 + hsi_identity
        sar = sar + sar_identity

        # Apply dropout if specified
        if self.dropout is not None:
            hsi_reshape = hsi.reshape(B, C * N, H, W)
            hsi_reshape = self.dropout(hsi_reshape)
            hsi = hsi_reshape.reshape(B, C, N, H, W)
            sar = self.dropout(sar)

        # Apply MSBlock for HSI
        hsi_reshape = hsi.reshape(B, C * N, H, W)
        hsi_reshape = self.MBh(hsi_reshape)
        
        # Reshape for output
        hsi = hsi_reshape.reshape(B, C, N, H, W)
        sar = self.MBx(sar)

        # Feature fusion with SSM attention
        out_hsi, out_sar = self.FSSBlock(hsi, sar)

        return out_hsi, out_sar


class Net(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        with open('dataset_info.yaml', 'r') as file:
            data = yaml.safe_load(file)
        data = data[dataset]

        # Dataset parameters
        self.out_features = data['num_classes']
        d_state = data['d_state']
        img_size = data['window_size']
        pca_num = data['pca_num']
        sar_num = data['slar_channel_num']
        
        # Channel configuration
        cha_h = [12, 24, 36, 48]
        cha_x = [96, 192, 288, 384]
        
        # Model hyperparameters
        expand = max(0.5, data.get('expand', 0.75))
        self.layer_num = 1
        channel_reduction = data.get('channel_reduction', 4)
        spatial_kernel_size = data.get('spatial_kernel_size', 7)
        dropout_rate = data.get('dropout_rate', 0.0)
        
        # Create the MCA-Mamba layer
        self.layers = nn.ModuleList()
        layer_dropout = dropout_rate * (1 + 0.2 * 0) if dropout_rate > 0 else 0
        self.layers.append(
            Syn_layer(1, cha_h[0], sar_num, cha_x[0], pca_num, img_size, d_state, 0, expand, 
                      channel_reduction, spatial_kernel_size, layer_dropout))

        # Update image dimensions after processing
        pca_num = pca_num - (9 - 2 * (0 + 1) + 1)
        img_size = img_size - 2
        
        # Calculate final fusion channels
        final_fusion_channels = cha_x[0] + cha_h[0] * pca_num
        final_reduction = max(2, int(channel_reduction / (self.layer_num + 1)))
        
        # Feature Fusion Attention (FFus-Attention)
        self.fusion_channel_attention = ChannelAttentionModule(final_fusion_channels, reduction=final_reduction)
        self.fusion_spatial_attention = SpatialAttentionModule(kernel_size=min(11, spatial_kernel_size + 2 * self.layer_num))
        
        # Multi-scale Feature Fusion (MFFus)
        self.multiscale_fusion = nn.Sequential(
            nn.Conv2d(final_fusion_channels, final_fusion_channels // 2, kernel_size=1),
            nn.BatchNorm2d(final_fusion_channels // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 1.5) if dropout_rate > 0 else nn.Identity()
        )
        
        # Global Context Attention (GCon-Attn)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_fusion_channels, final_fusion_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(final_fusion_channels // 4, final_fusion_channels),
            nn.Sigmoid()
        )
        
        # Classifier
        self.fusionlinear_fusion = nn.Sequential(
            nn.Linear(in_features=final_fusion_channels * (img_size ** 2) // 2, out_features=2048),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout_rate * 1.5) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(in_features=1024, out_features=self.out_features),
        )

    def forward(self, hsi, sar):
        # Feature extraction and initial fusion through the Syn_layer
        hsi, sar = self.layers[0](hsi, sar)
        
        # Reshape HSI features for fusion
        B, C, N, H, W = hsi.size()
        hsi = hsi.reshape(B, C * N, H, W)
        
        # Feature fusion - concatenate HSI and SAR features
        fusion_feat_m = torch.cat((hsi, sar), dim=1)
        
        # Apply global context attention
        if hasattr(self, 'global_context'):
            global_weights = self.global_context(fusion_feat_m)
            global_weights = global_weights.view(B, -1, 1, 1)
            fusion_feat_m = fusion_feat_m * global_weights
        
        # Apply feature fusion attention
        fusion_feat_m = fusion_feat_m * self.fusion_channel_attention(fusion_feat_m)
        fusion_feat_m = fusion_feat_m * self.fusion_spatial_attention(fusion_feat_m)
        
        # Multi-scale feature fusion
        fused_feat = self.multiscale_fusion(fusion_feat_m)
        
        # Reshape for classification
        B, C, H, W = fused_feat.size()
        fusion_feat = fused_feat.reshape(B, -1)
        
        # Final classification
        output_fusion = self.fusionlinear_fusion(fusion_feat)
        
        return fusion_feat_m, output_fusion