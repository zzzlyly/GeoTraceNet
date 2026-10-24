from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.decoders import EMCAD, channel_shuffle, gcd

from emcad.registry import DECODER_REGISTRY


_INHERITANCE_GAIN = 0.55
_LEARNED_STATE_GAIN = 0.10
_LEARNED_ROUTE_GAIN = 0.10
_DIRECT_ANCESTRY_GAIN = 0.15
_WRITE_CENTER = 0.050
_WRITE_RADIUS = 0.010
_REPAIR_FLOOR = 0.15
_EPS = 1e-6


def _smooth_channel_rms(value: torch.Tensor) -> torch.Tensor:
    """Zero-preserving channel RMS with finite gradients on a flat field."""
    return (
        torch.sqrt(value.float().square().mean(dim=1, keepdim=True) + _EPS**2)
        - _EPS
    )


def _symmetric_channel_agreement(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    numerator = 2.0 * (left.float() * right.float()).mean(dim=1, keepdim=True)
    denominator = (
        left.float().square().mean(dim=1, keepdim=True)
        + right.float().square().mean(dim=1, keepdim=True)
        + _EPS
    )
    return numerator / denominator


def _adjacent_skip_ancestry_replay(
    admitted_skip: torch.Tensor,
    replayed_skip: torch.Tensor,
    decoder_context: torch.Tensor,
    inherited_ancestry: torch.Tensor | None = None,
    learned_state: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute support and repair signals from admitted and replayed skips."""
    if admitted_skip.shape != replayed_skip.shape:
        raise ValueError("admitted and replayed skips must have matching shapes")
    if admitted_skip.ndim != 4 or decoder_context.ndim != 4:
        raise ValueError("ancestry replay tensors must have shape [B, C, H, W]")
    if (
        admitted_skip.shape[0] != decoder_context.shape[0]
        or admitted_skip.shape[-2:] != decoder_context.shape[-2:]
    ):
        raise ValueError("decoder context must share batch and spatial dimensions")

    admitted = admitted_skip.float()
    replayed = replayed_skip.float()
    context = decoder_context.float()
    scalar_shape = (admitted.shape[0], 1, admitted.shape[2], admitted.shape[3])
    if inherited_ancestry is None:
        inherited = admitted.new_zeros(scalar_shape)
        has_inheritance = False
    else:
        if inherited_ancestry.shape != scalar_shape:
            raise ValueError("inherited ancestry must have shape [B, 1, H, W]")
        inherited = inherited_ancestry.float()
        has_inheritance = True
    if learned_state is None:
        learned = admitted.new_zeros(scalar_shape)
    else:
        if learned_state.shape != scalar_shape:
            raise ValueError("learned state must have shape [B, 1, H, W]")
        learned = learned_state.float()

    ancestry_effect = admitted - replayed
    ancestry_support = _symmetric_channel_agreement(ancestry_effect, admitted)
    context_agreement = _symmetric_channel_agreement(context, admitted)
    admitted_scale = _smooth_channel_rms(admitted)
    replayed_scale = _smooth_channel_rms(replayed)
    effect_scale = _smooth_channel_rms(ancestry_effect)
    replay_gap = effect_scale / (admitted_scale + replayed_scale + _EPS)

    state_logits = (
        1.10 * ancestry_support
        + 0.35 * context_agreement
        + _LEARNED_STATE_GAIN * learned
    )
    if has_inheritance:
        state_logits = state_logits + _INHERITANCE_GAIN * inherited
    ancestry_state = torch.tanh(state_logits)

    # Blend admitted and replayed ancestry using the learned support state.
    repair_credit = _REPAIR_FLOOR + (1.0 - 2.0 * _REPAIR_FLOOR) * (
        0.5 - 0.5 * ancestry_state
    )
    ancestry_keep_credit = 1.0 - repair_credit
    replay_repair = replayed - admitted
    route_proposal = (
        repair_credit * replay_repair
        + _DIRECT_ANCESTRY_GAIN * ancestry_state * admitted
    )
    return {
        "ancestry_effect": ancestry_effect,
        "ancestry_support": ancestry_support,
        "context_agreement": context_agreement,
        "replay_gap": replay_gap,
        "inherited_ancestry": inherited,
        "ancestry_state": ancestry_state,
        "repair_credit": repair_credit,
        "ancestry_keep_credit": ancestry_keep_credit,
        "replay_repair": replay_repair,
        "route_proposal": route_proposal,
    }


class AdjacentSkipAncestryReplayEMCAD(EMCAD):
    """EMCAD decoder with leave-one-previous-skip ancestry replay."""

    def __init__(
        self,
        channels: list[int] = [512, 320, 128, 64],
        kernel_sizes: list[int] = [1, 3, 5],
        expansion_factor: int = 2,
        dw_parallel: bool = True,
        add: bool = True,
        lgag_ks: int = 3,
        activation: str = "relu6",
    ):
        if not add or not dw_parallel:
            raise ValueError(
                "adjacent ancestry replay requires additive parallel MSCB branches"
            )
        super().__init__(
            channels=channels,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            lgag_ks=lgag_ks,
            activation=activation,
        )
        transition_channels = channels[1:]
        self.ancestry_state_writers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=True),
                    nn.GELU(),
                    nn.Conv2d(8, 1, kernel_size=1, bias=True),
                )
                for _ in transition_channels
            ]
        )
        self.ancestry_state_bridges = nn.ModuleList(
            [nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False) for _ in range(2)]
        )
        for bridge in self.ancestry_state_bridges:
            nn.init.zeros_(bridge.weight)
            bridge.weight.data[0, 0, 1, 1] = 1.0
        self.ancestry_route_writers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channel,
                        channel,
                        kernel_size=3,
                        padding=1,
                        groups=channel,
                        bias=False,
                    ),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=1, bias=False),
                )
                for channel in transition_channels
            ]
        )
        self.write_strengths = nn.Parameter(torch.zeros(len(transition_channels)))
        self.latest_debug: dict[str, torch.Tensor] = {}
        self.latest_maps: dict[str, torch.Tensor] = {}
        self.latest_live: dict[str, torch.Tensor] = {}

    @staticmethod
    def _running_conv_norm(
        sequence: nn.Sequential,
        feature: torch.Tensor,
        *,
        activate: bool,
    ) -> torch.Tensor:
        convolved = sequence[0](feature)
        norm = sequence[1]
        normalized = F.batch_norm(
            convolved,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            momentum=0.0,
            eps=norm.eps,
        )
        return sequence[2](normalized) if activate else normalized

    @classmethod
    def _running_eucb_route(cls, eucb: nn.Module, parent: torch.Tensor) -> torch.Tensor:
        upsampled = eucb.up_dwc[0](parent)
        convolved = eucb.up_dwc[1](upsampled)
        norm = eucb.up_dwc[2]
        normalized = F.batch_norm(
            convolved,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            momentum=0.0,
            eps=norm.eps,
        )
        activated = eucb.up_dwc[3](normalized)
        return eucb.pwc(channel_shuffle(activated, eucb.in_channels))

    @classmethod
    def _running_lgag_route(
        cls,
        lgag: nn.Module,
        decoder_query: torch.Tensor,
        raw_skip: torch.Tensor,
    ) -> torch.Tensor:
        decoder_branch = cls._running_conv_norm(
            lgag.W_g, decoder_query, activate=False
        )
        skip_branch = cls._running_conv_norm(lgag.W_x, raw_skip, activate=False)
        activated = lgag.activation(decoder_branch + skip_branch)
        gate_logit = cls._running_conv_norm(lgag.psi, activated, activate=False)
        return raw_skip * lgag.psi[2](gate_logit)

    @classmethod
    def _running_mscb(cls, block: nn.Module, feature: torch.Tensor) -> torch.Tensor:
        expanded = cls._running_conv_norm(block.pconv1, feature, activate=True)
        branches = [
            cls._running_conv_norm(branch, expanded, activate=True)
            for branch in block.msdc.dwconvs
        ]
        combined = branches[0]
        for branch in branches[1:]:
            combined = combined + branch
        combined = channel_shuffle(
            combined, gcd(block.combined_channels, block.out_channels)
        )
        projected = cls._running_conv_norm(block.pconv2, combined, activate=False)
        return feature + projected

    def _refine(self, stage: int, feature: torch.Tensor) -> torch.Tensor:
        feature = getattr(self, f"cab{stage}")(feature) * feature
        feature = self.sab(feature) * feature
        return getattr(self, f"mscb{stage}")(feature)

    def _running_refine(self, stage: int, feature: torch.Tensor) -> torch.Tensor:
        feature = getattr(self, f"cab{stage}")(feature) * feature
        feature = self.sab(feature) * feature
        return self._running_mscb(getattr(self, f"mscb{stage}")[0], feature)

    def _inherit_ancestry(
        self,
        state: torch.Tensor,
        bridge: nn.Module,
        target: torch.Tensor,
    ) -> torch.Tensor:
        resized = F.interpolate(
            state,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.tanh(0.75 * bridge(resized.to(target.dtype)).float())

    def _ancestry_transition(
        self,
        index: int,
        stage: int,
        upsampled: torch.Tensor,
        admitted_skip: torch.Tensor,
        replayed_skip: torch.Tensor,
        inherited_ancestry: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(device_type=upsampled.device.type, enabled=False):
            preliminary = _adjacent_skip_ancestry_replay(
                admitted_skip.float(),
                replayed_skip.float(),
                upsampled.float(),
                None
                if inherited_ancestry is None
                else inherited_ancestry.float(),
            )
            state_input = torch.cat(
                [
                    preliminary["ancestry_support"],
                    preliminary["replay_gap"],
                    preliminary["context_agreement"],
                    preliminary["inherited_ancestry"],
                ],
                dim=1,
            )
        learned_state = self.ancestry_state_writers[index](
            state_input.to(upsampled.dtype)
        )

        with torch.autocast(device_type=upsampled.device.type, enabled=False):
            decision = _adjacent_skip_ancestry_replay(
                admitted_skip.float(),
                replayed_skip.float(),
                upsampled.float(),
                None
                if inherited_ancestry is None
                else inherited_ancestry.float(),
                learned_state.float(),
            )
            direct_proposal = decision["route_proposal"]
            fused_raw = upsampled.float() + admitted_skip.float()
            write_scale = (
                _smooth_channel_rms(fused_raw)
                + 0.25 * _smooth_channel_rms(direct_proposal)
                + _EPS
            )

        learned_route = self.ancestry_route_writers[index](
            direct_proposal.to(upsampled.dtype)
        ).float()
        proposal = direct_proposal + _LEARNED_ROUTE_GAIN * write_scale * torch.tanh(
            learned_route / write_scale
        )
        strength = _WRITE_CENTER + _WRITE_RADIUS * torch.tanh(
            self.write_strengths[index].float()
        )
        correction = strength * write_scale * torch.tanh(proposal / write_scale)
        routed_skip = admitted_skip.float() + correction
        state = decision["ancestry_state"]

        metrics = {
            "adjacent_ancestry_support": decision["ancestry_support"].mean(
                dim=(1, 2, 3)
            ),
            "adjacent_replay_gap": decision["replay_gap"].mean(dim=(1, 2, 3)),
            "ancestry_keep_credit": decision["ancestry_keep_credit"].mean(
                dim=(1, 2, 3)
            ),
            "ancestry_state_magnitude": state.abs().mean(dim=(1, 2, 3)),
            "inherited_ancestry_magnitude": (
                torch.zeros(upsampled.shape[0], device=upsampled.device)
                if inherited_ancestry is None
                else inherited_ancestry.float().abs().mean(dim=(1, 2, 3))
            ),
            "ancestry_write_displacement": correction.abs().mean(dim=(1, 2, 3)),
            "ancestry_write_ratio": correction.abs().mean(dim=(1, 2, 3))
            / (write_scale.mean(dim=(1, 2, 3)) + _EPS),
        }
        self.latest_live[f"ancestry_state_stage{stage}"] = state
        self.latest_maps[f"admitted_skip_stage{stage}"] = admitted_skip.detach()
        self.latest_maps[f"replayed_skip_stage{stage}"] = replayed_skip.detach()
        self.latest_maps[f"ancestry_effect_stage{stage}"] = decision[
            "ancestry_effect"
        ].detach()
        self.latest_maps[f"ancestry_support_stage{stage}"] = decision[
            "ancestry_support"
        ].detach()
        self.latest_maps[f"ancestry_state_stage{stage}"] = state.detach()
        self.latest_maps[f"ancestry_correction_stage{stage}"] = correction.detach()
        return routed_skip.to(admitted_skip.dtype), state.to(admitted_skip.dtype), metrics

    def forward(self, x, skips):
        # Preserve the complete inherited coarse/pretrained decoder stage.
        d4 = self._refine(4, x)
        decoded = [d4]
        previous_without_skip: torch.Tensor | None = None
        inherited: torch.Tensor | None = None
        metrics_by_stage: list[dict[str, torch.Tensor]] = []
        self.latest_maps = {}
        self.latest_live = {}

        for index, (stage, raw_skip) in enumerate(zip((3, 2, 1), skips)):
            upsampled = getattr(self, f"eucb{stage}")(decoded[-1])
            admitted = getattr(self, f"lgag{stage}")(g=upsampled, x=raw_skip)
            if previous_without_skip is None:
                replayed = admitted
            else:
                replay_up = self._running_eucb_route(
                    getattr(self, f"eucb{stage}"), previous_without_skip
                )
                replayed = self._running_lgag_route(
                    getattr(self, f"lgag{stage}"), replay_up, raw_skip
                )
            if inherited is not None:
                inherited = self._inherit_ancestry(
                    inherited,
                    self.ancestry_state_bridges[index - 1],
                    upsampled,
                )
            routed, inherited, metrics = self._ancestry_transition(
                index, stage, upsampled, admitted, replayed, inherited
            )

            # The alternate parent omits only the current skip from the actual parent.
            # Running BN statistics prevent this counterfactual from updating the
            # standard path's normalization state a second time.
            previous_without_skip = self._running_refine(stage, upsampled)
            decoded.append(self._refine(stage, upsampled + routed))
            metrics_by_stage.append(metrics)

        self.latest_debug = {
            key: torch.stack([metrics[key] for metrics in metrics_by_stage], dim=1)
            for key in metrics_by_stage[0]
        }
        return decoded


@DECODER_REGISTRY.register("adjacent_skip_ancestry_replay_emcad")
def build_adjacent_skip_ancestry_replay_decoder(
    *, channels: list[int], config: dict
):
    return AdjacentSkipAncestryReplayEMCAD(
        channels=channels,
        kernel_sizes=config.get("kernel_sizes", [1, 3, 5]),
        expansion_factor=config.get("expansion_factor", 2),
        dw_parallel=config.get("dw_parallel", True),
        add=config.get("add", True),
        lgag_ks=config.get("lgag_ks", 3),
        activation=config.get("activation", "relu6"),
    )
