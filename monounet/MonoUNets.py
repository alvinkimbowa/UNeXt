import torch
import torch.nn as nn

from monounet.mono_layer import Mono2DV3

__all__ = [
    'MonoUNetBase',
    'MonoUNetE1',
    'MonoUNetE12',
    'MonoUNetE123',
    'MonoUNetE1234',
    # 'MonoUNetE1234D1',
    ## Only run the next two if the previous one is successful.
    # 'MonoUNetE1234D12',
    # 'MonoUNetE1234D123',
    ## Run after experiments above with best configuration.
    # 'MonoUNetS2',
    # 'MonoUNetS4',
    ## Optional run to show saturation
    # 'MonoUNetS6',
]

# -------------------------
# Baseline Lean UNet
# -------------------------

class MonoUNetBase(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        img_size=(256, 256),
        init_filters=2,
        max_filters=2,
        deep_supervision=True,
        encoder_cls=None,
        encoder_kwargs=None,
    ):
        super().__init__()

        num_stages = 7 if max(img_size[0], img_size[1]) <= 256 else 8
        filters = [min(max_filters, init_filters * 2**i) for i in range(num_stages)]
        encoder_kwargs = encoder_kwargs or {}

        if encoder_cls is None:
            self.encoder = XTinyEncoder(in_channels, filters, deep_supervision=deep_supervision)
        else:
            self.encoder = encoder_cls(in_channels, filters, deep_supervision=deep_supervision, **encoder_kwargs)

        self.decoder = XTinyDecoder(self.encoder, num_classes, filters, deep_supervision)
        self.initialize_weights()

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    m.bias = nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    m.weight = nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias = nn.init.constant_(m.bias, 0)


class XTinyEncoder(nn.Module):
    def __init__(self, in_channels=1, filters=None, deep_supervision=True):
        super().__init__()
        filters = filters or []

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(filters[0], eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        stages = []
        pooling_layers = []

        num_stages = len(filters)
        for i in range(num_stages):
            block = nn.Sequential(
                nn.Conv2d(filters[i], filters[i], kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(filters[i], eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            stages.append(nn.Sequential(block, block))

            if i < num_stages - 1:
                pooling_layers.append(nn.Conv2d(filters[i], filters[i + 1], kernel_size=3, stride=2, padding=1))

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
    def __init__(self, encoder, num_classes=2, filters=None, deep_supervision=True):
        super().__init__()
        filters = filters or []
        self.deep_supervision = deep_supervision

        stages = []
        transpose_convs = []
        ds_seg_heads = []

        num_stages = len(encoder.stages)
        for i in range(1, num_stages):
            stages.append(
                nn.Sequential(
                    nn.Conv2d(filters[-(i + 1)] * 2, filters[-(i + 1)], kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(filters[-(i + 1)], eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                )
            )
            transpose_convs.append(nn.ConvTranspose2d(filters[-i], filters[-(i + 1)], kernel_size=2, stride=2))
            ds_seg_heads.append(nn.Conv2d(filters[-(i + 1)], num_classes, kernel_size=1, stride=1))

        self.stages = nn.ModuleList(stages)
        self.transpose_convs = nn.ModuleList(transpose_convs)
        self.ds_seg_heads = nn.ModuleList(ds_seg_heads)

    def forward(self, x, skip_connections):
        outputs = []
        for i in range(len(self.stages)):
            x = self.transpose_convs[i](x)
            x = torch.cat([x, skip_connections[-(i + 1)]], dim=1)
            x = self.stages[i](x)
            outputs.append(self.ds_seg_heads[i](x))
        return outputs if self.deep_supervision else outputs[-1]


# -------------------------------------------
# Monogenic encoder with "inject up to stage"
# -------------------------------------------

class MonoUNetEEncoder(XTinyEncoder):
    """
    Inject monogenic features into encoder stages cumulatively:
    inject_upto = 1 -> stage 0 only
    inject_upto = 2 -> stages 0..1
    inject_upto = 3 -> stages 0..2
    inject_upto = 4 -> stages 0..3
    """
    def __init__(self, in_channels=1, filters=None, deep_supervision=True,
                 inject_upto: int = 1, gate_encoder: bool = True,
                 nscale: int | None = None):
        super().__init__(in_channels, filters, deep_supervision=deep_supervision)
        filters = filters or []
        num_stages = len(filters)

        assert 1 <= inject_upto <= num_stages, f"inject_upto must be in [1,{num_stages}]"
        self.inject_upto = inject_upto
        self.gate_encoder = gate_encoder

        self.mono2d = Mono2DV3(in_channels, nscale=nscale, norm="std", return_phase=True)

        # Build LP pyramid progressively (cheap + consistent)
        self.lp_down = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2, stride=2)
            for _ in range(num_stages - 1)
        ])

        # Project monogenic features to match stage width (C_i)
        self.lp_proj = nn.ModuleList([
            nn.Conv2d(self.mono2d.out_channels, filters[i], kernel_size=1, stride=1, bias=False)
            for i in range(num_stages)
        ])

        if self.gate_encoder:
            # A very simple, stable residual gate per stage:
            # x <- x + sigma(a_i) * lp
            self.gate_a = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_stages)])

        # Replace pooling with your norm+act version (since your MonoUNetE1Encoder did)
        self.pooling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(filters[i], filters[i+1], kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(filters[i+1], eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            for i in range(num_stages - 1)
        ])

    def _inject(self, x, lp_stage, i: int):
        if not self.gate_encoder:
            return x + lp_stage
        alpha = torch.sigmoid(self.gate_a[i])  # scalar in (0,1)
        return x + alpha * lp_stage

    def forward(self, x):
        lp_feat = self.mono2d(x)  # B x Cmono x H x W
        x = self.stem(x)          # B x C0 x H x W

        skip_connections = []
        for i, stage in enumerate(self.stages):
            if i < self.inject_upto:
                lp_stage = self.lp_proj[i](lp_feat)  # -> B x C_i x H_i x W_i
                x = self._inject(x, lp_stage, i)

            x = stage(x)

            if i < len(self.stages) - 1:
                skip_connections.append(x)
                x = self.pooling_layers[i](x)

                # progressive downsample of monogenic features to match next stage
                lp_feat = self.lp_down[i](lp_feat)

        return x, skip_connections


# -------------------------
# MonoUNetE variants - Monogenic features injected into encoder stages
# -------------------------

class MonoUNetE1(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"inject_upto": 1, "nscale": 3, "gate_encoder": False}, **kwargs)

class MonoUNetE12(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"inject_upto": 2, "nscale": 3, "gate_encoder": False}, **kwargs)

class MonoUNetE123(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"inject_upto": 3, "nscale": 3, "gate_encoder": False}, **kwargs)

class MonoUNetE1234(MonoUNetBase):
    def __init__(self, **kwargs):
        super().__init__(encoder_cls=MonoUNetEEncoder, encoder_kwargs={"inject_upto": 4, "nscale": 3, "gate_encoder": False}, **kwargs)
