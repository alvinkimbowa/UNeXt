import torch
import torch.nn as nn

from monounet.mono_layer import Mono2D, Mono2DV2

__all__ = ['XTinyUNet', 'XTinyMonoUNetScale1', 'XTinyMonoUNetScale6', 'XTinyMonoV2UNetScale1', 'XTinyMonoV2UNetScale6', 'XTinyMonoV2GatedUNet']


class XTinyUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super(XTinyUNet, self).__init__()
        
        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        
        self.encoder = XTinyEncoder(in_channels, filters, deep_supervision=deep_supervision)
        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                m.bias = nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight = nn.init.constant_(m.weight, 1)
                m.bias = nn.init.constant_(m.bias, 0)


class XTinyEncoder(nn.Module):
    def __init__(self, in_channels=1, filters=[], deep_supervision=True):
        super(XTinyEncoder, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(filters[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        stages = []
        pooling_layers = []

        num_stages = len(filters)
        for i in range(num_stages):            
            # Conv Block
            block = nn.Sequential(
                nn.Conv2d(filters[i], filters[i], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(filters[i], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
            stages.append(nn.Sequential(block, block))

            if i < num_stages - 1:
                # Pooling Block
                pooling_layers.append(nn.Conv2d(filters[i], filters[i+1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        
        self.stages = nn.ModuleList(stages)
        self.pooling_layers = nn.ModuleList(pooling_layers)

    def forward(self, x):
        x = self.stem(x)
        skip_connections = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                skip_connections.append(x)
                x = self.pooling_layers[i](x)
        return x, skip_connections


class XTinyDecoder(nn.Module):
    def __init__(self, encoder, num_classes=2, filters=[], deep_supervision=True):
        super(XTinyDecoder, self).__init__()
        
        self.deep_supervision = deep_supervision
        stages = []
        num_stages = len(encoder.stages)
        transpose_convs = []
        ds_seg_heads = []
        for i in range(1, num_stages):
            stages.append(nn.Sequential(
                nn.Conv2d(filters[-(i+1)]*2, filters[-(i+1)], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(filters[-(i+1)], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            ))

            transpose_convs.append(nn.ConvTranspose2d(filters[-i], filters[-(i+1)], kernel_size=2, stride=2))
            
            ds_seg_heads.append(nn.Sequential(
                nn.Conv2d(filters[-(i+1)], num_classes, kernel_size=1, stride=1)
            ))
        
        self.stages = nn.ModuleList(stages)
        self.transpose_convs = nn.ModuleList(transpose_convs)
        self.ds_seg_heads = nn.ModuleList(ds_seg_heads)

    def forward(self, x, skip_connections):
        outputs = []
        for i in range(len(self.stages)):
            x = self.transpose_convs[i](x)
            x = torch.cat([x, skip_connections[-(i+1)]], dim=1)
            x = self.stages[i](x)
            outputs.append(self.ds_seg_heads[i](x))
        
        if self.deep_supervision:
            return outputs
        else:
            return outputs[-1]


class XTinyMonoUNetScale1(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        self.mono2d = Mono2D(in_channels, nscale=1, norm="std", return_phase=True)
        
    def forward(self, x):
        x = self.mono2d(x)
        return super().forward(x)


class XTinyMonoUNetScale6(XTinyMonoUNetScale1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        self.mono2d = Mono2D(in_channels, nscale=6, norm="std", return_phase=True)


class XTinyMonoV2UNetScale1(XTinyMonoUNetScale1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        print("in_channels", in_channels)
        self.mono2d = Mono2DV2(in_channels, nscale=1, norm="std", return_phase=True)
        in_channels = self.mono2d.out_channels
        print("in_channels after Mono2DV2", in_channels)
        
        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        
        self.encoder = XTinyEncoder(in_channels, filters, deep_supervision=deep_supervision)
        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()


class XTinyMonoV2UNetScale6(XTinyMonoUNetScale1):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        print("in_channels", in_channels)
        self.mono2d = Mono2DV2(in_channels, nscale=6, norm="std", return_phase=True)
        in_channels = self.mono2d.out_channels
        print("in_channels after Mono2DV2", in_channels)
        
        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        
        self.encoder = XTinyEncoder(in_channels, filters, deep_supervision=deep_supervision)
        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)

        self.initialize_weights()

class XTinyMonoV2GatedUNet(XTinyUNet):
    def __init__(self, in_channels=1, num_classes=2, img_size=(256, 256), init_filters=1, max_filters=2, deep_supervision=True):
        super().__init__(in_channels, num_classes, img_size, init_filters, max_filters, deep_supervision)
        self.mono2d = Mono2DV2(in_channels, nscale=1, norm="std", return_phase=True)
        
        # Learnable affine for gate and residual strength
        self.gate_weight = nn.Parameter(torch.ones(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.randn(1))
        print("\n\nUsing MonoGatedUNet with mono2d layer\n\n")
    
    def forward(self, x):
        mono_feat = self.mono2d(x)
        gate = torch.sigmoid(self.gate_weight * mono_feat + self.gate_bias)
        x = x * (1 + self.alpha * gate)     # Use a residual connection to stabilize training
        return super().forward(x)


if __name__ == "__main__":
    x = torch.randn((1,1,256,256))
    model = XTinyUNet(in_channels=1, num_classes=2, img_size=(256, 256), deep_supervision=True)
    print("Sample deep supervision outputs")
    for output in model(x):
        print(output.shape)
    
    # total params
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")