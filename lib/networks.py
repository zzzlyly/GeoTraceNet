import torch
import torch.nn as nn

from emcad.models.components import build_named_backbone  # noqa: F401
from emcad.models.stage_config import build_legacy_compatible_arch
from emcad.registry import BACKBONE_REGISTRY, DECODER_REGISTRY, FLOW_REGISTRY, HEAD_REGISTRY, STEM_REGISTRY


class ModularEMCADNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        kernel_sizes=[1, 3, 5],
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        lgag_ks=3,
        activation="relu",
        encoder="pvt_v2_b2",
        pretrain=True,
        pretrained_dir="./pretrained_pth/pvt/",
        arch=None,
    ):
        super().__init__()
        self.model_config = {
            "num_classes": num_classes,
            "kernel_sizes": kernel_sizes,
            "expansion_factor": expansion_factor,
            "dw_parallel": dw_parallel,
            "add": add,
            "lgag_ks": lgag_ks,
            "activation": activation,
            "encoder": encoder,
            "pretrain": pretrain,
            "pretrained_dir": pretrained_dir,
            "arch": arch or {},
        }
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.lgag_ks = lgag_ks
        self.activation = activation
        self.encoder_name = encoder
        self.pretrain = pretrain
        self.pretrained_dir = pretrained_dir

        self.arch_config = build_legacy_compatible_arch(
            {
                "num_classes": num_classes,
                "kernel_sizes": kernel_sizes,
                "expansion_factor": expansion_factor,
                "dw_parallel": dw_parallel,
                "add": add,
                "lgag_ks": lgag_ks,
                "activation": activation,
                "encoder": encoder,
                "pretrain": pretrain,
                "pretrained_dir": pretrained_dir,
                "arch": arch or {},
            }
        )
        self.conv = self.build_stem()
        self.backbone, self.channels, self.backbone_pretrained_path = self.build_backbone()
        self.load_backbone_pretrain_if_needed()
        self.log_module_params(f"{self.encoder_name} backbone", self.backbone)

        self.decoder = self.build_decoder()
        self.log_module_params("EMCAD decoder", self.decoder)

        self.heads = self.build_heads()
        self._sync_head_aliases()

    def build_stem(self):
        stem_cfg = dict(self.arch_config["stem"])
        stem_name = stem_cfg.pop("name", "grayscale_to_rgb")
        return STEM_REGISTRY.build(stem_name, config=stem_cfg)

    def resolve_backbone_name(self):
        backbone_name = self.arch_config["backbone"].get("name", self.encoder_name)
        if backbone_name not in BACKBONE_REGISTRY.keys():
            print("Encoder not implemented! Continuing with default encoder pvt_v2_b2.")
            backbone_name = "pvt_v2_b2"
            self.arch_config["backbone"]["name"] = backbone_name
        self.encoder_name = backbone_name
        return backbone_name

    def build_backbone(self):
        backbone_name = self.resolve_backbone_name()
        backbone_cfg = dict(self.arch_config["backbone"])
        backbone_cfg["name"] = backbone_name
        backbone_cfg.setdefault("pretrain", self.pretrain)
        backbone_cfg.setdefault("pretrained_dir", self.pretrained_dir)
        adapter = BACKBONE_REGISTRY.build(backbone_name, config=backbone_cfg)
        return adapter.module, list(adapter.channels), adapter.pretrained_path

    def load_backbone_pretrain_if_needed(self):
        if not (self.pretrain and "pvt_v2" in self.encoder_name and self.backbone_pretrained_path):
            return
        save_model = torch.load(self.backbone_pretrained_path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

    def build_decoder(self):
        decoder_cfg = dict(self.arch_config["decoder"])
        decoder_name = decoder_cfg.pop("name", "emcad")
        return DECODER_REGISTRY.build(decoder_name, channels=self.channels, config=decoder_cfg)

    def build_heads(self):
        head_cfg = dict(self.arch_config["heads"])
        head_name = head_cfg.pop("name", "conv_4scale")
        head_cfg.setdefault("num_classes", self.num_classes)
        return HEAD_REGISTRY.build(head_name, channels=self.channels, config=head_cfg)

    def _sync_head_aliases(self):
        self.out_head4 = self.heads[0]
        self.out_head3 = self.heads[1]
        self.out_head2 = self.heads[2]
        self.out_head1 = self.heads[3]

    def set_heads(self, heads):
        self.heads = nn.ModuleList(list(heads))
        self._sync_head_aliases()

    def get_prediction_heads(self):
        return [self.out_head4, self.out_head3, self.out_head2, self.out_head1]

    def log_module_params(self, name, module):
        print("Model %s created, param count: %d" % (name + ": ", sum(m.numel() for m in module.parameters())))

    def preprocess_input(self, x):
        if x.size()[1] == 1:
            x = self.conv(x)
        return x

    def encode(self, x):
        return self.flow.encode(self, x)

    def decode(self, encoded):
        return self.flow.decode(self, encoded)

    def predict(self, decoded):
        return self.flow.predict(self, decoded)

    def upsample_outputs(self, predictions):
        return self.flow.upsample(self, predictions)

    def forward_features(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        predictions = self.predict(decoded)
        return {
            "encoded": encoded,
            "decoded": decoded,
            "predictions": predictions,
        }

    def forward(self, x, mode="test"):
        features = self.forward_features(x)
        outputs = self.upsample_outputs(features["predictions"])
        if mode == "test":
            return outputs
        return outputs


class EMCADNet(ModularEMCADNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flow_cfg = dict(self.arch_config["flow"])
        flow_name = flow_cfg.pop("name", "standard_emcad")
        self.flow = FLOW_REGISTRY.build(flow_name, config=flow_cfg)


if __name__ == "__main__":
    model = EMCADNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    P = model(input_tensor)
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())
