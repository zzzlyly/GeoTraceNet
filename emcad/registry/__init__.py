from emcad.registry.base import Registry


MODEL_REGISTRY = Registry("models")
STEM_REGISTRY = Registry("stems")
BACKBONE_REGISTRY = Registry("backbones")
DECODER_REGISTRY = Registry("decoders")
HEAD_REGISTRY = Registry("heads")
FLOW_REGISTRY = Registry("flows")
DATASET_REGISTRY = Registry("datasets")
LOSS_REGISTRY = Registry("losses")
METRIC_REGISTRY = Registry("metrics")
HOOK_REGISTRY = Registry("hooks")


__all__ = [
    "Registry",
    "MODEL_REGISTRY",
    "STEM_REGISTRY",
    "BACKBONE_REGISTRY",
    "DECODER_REGISTRY",
    "HEAD_REGISTRY",
    "FLOW_REGISTRY",
    "DATASET_REGISTRY",
    "LOSS_REGISTRY",
    "METRIC_REGISTRY",
    "HOOK_REGISTRY",
]
