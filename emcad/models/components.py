from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.decoders import EMCAD
from lib.pvtv2 import pvt_v2_b2

from emcad.registry import BACKBONE_REGISTRY, DECODER_REGISTRY, FLOW_REGISTRY, HEAD_REGISTRY, STEM_REGISTRY


BACKBONE_SPECS = {
    "pvt_v2_b2": {"builder": pvt_v2_b2, "channels": [512, 320, 128, 64], "pretrained_file": "pvt_v2_b2.pth"},
}


class BackboneAdapter(nn.Module):
    def __init__(self, module: nn.Module, channels: list[int], pretrained_path: str | None = None):
        super().__init__()
        self.module = module
        self.channels = channels
        self.pretrained_path = pretrained_path

    def forward(self, x):
        return self.module(x)


class StandardEmcadFlow:
    def encode(self, model, x):
        x = model.preprocess_input(x)
        x1, x2, x3, x4 = model.backbone(x)
        return x1, x2, x3, x4

    def decode(self, model, encoded):
        x1, x2, x3, x4 = encoded
        return model.decoder(x4, [x3, x2, x1])

    def predict(self, model, decoded):
        return [head(feat) for head, feat in zip(model.get_prediction_heads(), decoded)]

    def upsample(self, model, predictions):
        scales = [32, 16, 8, 4]
        return [
            F.interpolate(pred, scale_factor=scale, mode="bilinear")
            for pred, scale in zip(predictions, scales)
        ]


@STEM_REGISTRY.register("grayscale_to_rgb")
def build_grayscale_to_rgb_stem(*, config: dict):
    return nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=1),
        nn.BatchNorm2d(3),
        nn.ReLU(inplace=True),
    )


@BACKBONE_REGISTRY.register("pvt_v2_b2")
def build_named_backbone(*, config: dict):
    name = config["name"]
    spec = BACKBONE_SPECS[name]
    pretrain = config.get("pretrain", True)
    pretrained_dir = config.get("pretrained_dir", "./pretrained_pth/pvt")
    module = spec["builder"]()
    pretrained_path = f"{pretrained_dir}/{spec['pretrained_file']}"
    return BackboneAdapter(module=module, channels=list(spec["channels"]), pretrained_path=pretrained_path)


@DECODER_REGISTRY.register("emcad")
def build_emcad_decoder(*, channels: list[int], config: dict):
    return EMCAD(
        channels=channels,
        kernel_sizes=config.get("kernel_sizes", [1, 3, 5]),
        expansion_factor=config.get("expansion_factor", 2),
        dw_parallel=config.get("dw_parallel", True),
        add=config.get("add", True),
        lgag_ks=config.get("lgag_ks", 3),
        activation=config.get("activation", "relu6"),
    )


@HEAD_REGISTRY.register("conv_4scale")
def build_conv_4scale_heads(*, channels: list[int], config: dict):
    num_classes = config.get("num_classes", 1)
    return nn.ModuleList([nn.Conv2d(channel, num_classes, 1) for channel in channels])


@FLOW_REGISTRY.register("standard_emcad")
def build_standard_emcad_flow(*, config: dict):
    return StandardEmcadFlow()
