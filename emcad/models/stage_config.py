from __future__ import annotations

from copy import deepcopy

from emcad.utils.dicts import deep_merge


def build_legacy_compatible_arch(model_cfg: dict) -> dict:
    defaults = {
        "stem": {"name": "grayscale_to_rgb"},
        "backbone": {
            "name": model_cfg.get("encoder", "pvt_v2_b2"),
            "pretrain": model_cfg.get("pretrain", True),
            "pretrained_dir": model_cfg.get("pretrained_dir", "./pretrained_pth/pvt"),
        },
        "decoder": {
            "name": "emcad",
            "kernel_sizes": model_cfg.get("kernel_sizes", [1, 3, 5]),
            "expansion_factor": model_cfg.get("expansion_factor", 2),
            "dw_parallel": model_cfg.get("dw_parallel", True),
            "add": model_cfg.get("add", True),
            "lgag_ks": model_cfg.get("lgag_ks", 3),
            "activation": model_cfg.get("activation", "relu6"),
        },
        "heads": {
            "name": "conv_4scale",
            "num_classes": model_cfg.get("num_classes", 1),
        },
        "flow": {"name": "standard_emcad"},
    }
    return deep_merge(defaults, deepcopy(model_cfg.get("arch", {})))
