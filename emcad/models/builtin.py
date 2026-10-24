from __future__ import annotations

from lib.networks import EMCADNet

from emcad.registry import MODEL_REGISTRY
from emcad.models import components as builtin_components  # noqa: F401


@MODEL_REGISTRY.register("emcad_seg")
def build_emcad_segmentation_model(*, config: dict):
    model_cfg = config["model"]
    return EMCADNet(
        num_classes=model_cfg.get("num_classes", 1),
        kernel_sizes=model_cfg.get("kernel_sizes", [1, 3, 5]),
        expansion_factor=model_cfg.get("expansion_factor", 2),
        dw_parallel=model_cfg.get("dw_parallel", True),
        add=model_cfg.get("add", True),
        lgag_ks=model_cfg.get("lgag_ks", 3),
        activation=model_cfg.get("activation", "relu6"),
        encoder=model_cfg.get("encoder", "pvt_v2_b2"),
        pretrain=model_cfg.get("pretrain", True),
        pretrained_dir=model_cfg.get("pretrained_dir", "./pretrained_pth/pvt"),
        arch=model_cfg.get("arch"),
    )
