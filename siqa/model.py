import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t


class SiameseSemanticIQA(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        structure_backbone: str = "",
        ablation_mode: str = "full",
        swin_name: str = "swin_tiny_patch4_window7_224",
        clip_name: str = "clip_vit_b32",
        freeze_backbones: bool = True,
        swin_local_path: str = "",
        clip_local_dir: str = "",
        clip_local_files_only: bool = False,
        clip_interpolate_pos_encoding: bool = True,
        clip_mult_enabled: bool = True,
        clip_mult_replace_raw: bool = True,
        clip_mult_l2_norm: bool = True,
        bottleneck_dim: int = 256,
        bottleneck_dropout: float = 0.5,
        semantic_gate_enabled: bool = True,
        semantic_gate_threshold: float = 0.4,
        semantic_gate_high_threshold: float = 0.5,
        semantic_gate_mode: str = "hard",
        gate_logit_strength: float = 12.0,
        soft_gate_logit_strength: float = 6.0,
    ):
        super().__init__()

        self.structure_backbone_name = structure_backbone or swin_name
        self.ablation_mode = str(ablation_mode).lower()
        self.clip_name = clip_name
        self.clip_interpolate_pos_encoding = bool(clip_interpolate_pos_encoding)
        self.clip_mult_enabled = bool(clip_mult_enabled)
        self.clip_mult_replace_raw = bool(clip_mult_replace_raw)
        self.clip_mult_l2_norm = bool(clip_mult_l2_norm)
        self.semantic_gate_enabled = bool(semantic_gate_enabled)
        self.semantic_gate_threshold = float(semantic_gate_threshold)
        self.semantic_gate_high_threshold = float(semantic_gate_high_threshold)
        self.semantic_gate_mode = str(semantic_gate_mode).lower()
        self.gate_logit_strength = float(gate_logit_strength)
        self.soft_gate_logit_strength = float(soft_gate_logit_strength)

        if self.ablation_mode not in {"full", "clip_only", "structure_only"}:
            raise ValueError("ablation_mode must be one of {'full','clip_only','structure_only'}")

        self.structure_backbone, self.structure_feature_dim, self.structure_backbone_type = self._build_structure_backbone(
            self.structure_backbone_name,
            swin_local_path=swin_local_path,
        )
        self.clip_backbone, self.clip_feature_dim = self._build_clip_backbone(
            clip_name,
            clip_local_dir=clip_local_dir,
            clip_local_files_only=clip_local_files_only,
        )

        if freeze_backbones:
            for param in self.structure_backbone.parameters():
                param.requires_grad = False
            for param in self.clip_backbone.parameters():
                param.requires_grad = False

        clip_branch_dim = self.clip_feature_dim * (3 if self.clip_mult_replace_raw else 4)
        fusion_dim = self.structure_feature_dim * 3 + clip_branch_dim + 1
        self.bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.SiLU(),
            nn.Dropout(bottleneck_dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )
        self.register_buffer("score_weights", torch.arange(0, num_classes, dtype=torch.float32))

    def _resolve_clip_model_id(self, clip_name: str) -> str:
        alias_map = {
            "clip_vit_b32": "openai/clip-vit-base-patch32",
            "clip_vit_l14": "openai/clip-vit-large-patch14",
            "clip_vit_l14_336": "openai/clip-vit-large-patch14-336",
        }
        if clip_name in alias_map:
            return alias_map[clip_name]
        if "/" in clip_name:
            return clip_name
        raise ValueError(
            "Unsupported CLIP backbone alias. Use one of "
            "{'clip_vit_b32','clip_vit_l14','clip_vit_l14_336'} or a HuggingFace model id."
        )

    def _load_swin_local_state_dict(self, model: nn.Module, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Swin local checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            elif "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid Swin checkpoint format: {path}")
        cleaned: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            cleaned[key] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            raise RuntimeError(f"Swin checkpoint missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            raise RuntimeError(f"Swin checkpoint unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    def _build_swin_backbone(self, swin_name: str, swin_local_path: str = ""):
        if swin_name != "swin_tiny_patch4_window7_224":
            raise ValueError("Currently only 'swin_tiny_patch4_window7_224' is supported for Swin backbone")

        if swin_local_path:
            model = swin_t(weights=None)
            self._load_swin_local_state_dict(model, swin_local_path)
            return model, int(model.head.in_features)

        try:
            model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load pretrained Swin-T from torchvision. "
                "For servers in China, pre-download the Swin checkpoint and set model.swin_local_path."
            ) from exc
        return model, int(model.head.in_features)

    def _build_dinov3_backbone(self, backbone_name: str):
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is required for DINOv3 backbone") from exc

        available = set(timm.list_models("*dinov3*"))
        if backbone_name not in available:
            raise ValueError(
                f"Unsupported DINOv3 backbone '{backbone_name}'. "
                f"Try one of: {sorted(available)}"
            )

        try:
            model = timm.create_model(backbone_name, pretrained=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load DINOv3 backbone '{backbone_name}' from timm. "
                "If you are in mainland China, set HF_ENDPOINT=https://hf-mirror.com "
                "before running to speed up HuggingFace downloads."
            ) from exc

        feature_dim = getattr(model, "num_features", None)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise RuntimeError(f"Unable to resolve feature dim for DINOv3 backbone '{backbone_name}'")
        return model, feature_dim

    def _build_structure_backbone(self, backbone_name: str, swin_local_path: str = ""):
        if backbone_name == "swin_tiny_patch4_window7_224":
            model, feature_dim = self._build_swin_backbone(backbone_name, swin_local_path=swin_local_path)
            return model, feature_dim, "swin"
        if "dinov3" in backbone_name:
            model, feature_dim = self._build_dinov3_backbone(backbone_name)
            return model, feature_dim, "dinov3"
        raise ValueError(
            "Unsupported structure backbone. Use 'swin_tiny_patch4_window7_224' "
            "or a timm DINOv3 model name like 'vit_base_patch16_dinov3'."
        )

    def _build_clip_backbone(self, clip_name: str, clip_local_dir: str = "", clip_local_files_only: bool = False):
        try:
            from transformers import CLIPVisionModel
        except ImportError as exc:
            raise RuntimeError("transformers is required for CLIP-ViT backbone") from exc

        if clip_local_dir and os.path.isdir(clip_local_dir):
            model_id = clip_local_dir
            local_files_only = True
        else:
            model_id = self._resolve_clip_model_id(clip_name)
            local_files_only = bool(clip_local_files_only)
        try:
            model = CLIPVisionModel.from_pretrained(model_id, local_files_only=local_files_only)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CLIP backbone '{model_id}'. "
                "This usually happens when HuggingFace is unreachable. "
                "If you are in mainland China, set HF_ENDPOINT=https://hf-mirror.com "
                "or set model.clip_local_dir to a local downloaded model path. "
                f"Underlying error: {exc}"
            ) from exc
        feature_dim = int(model.config.hidden_size)
        return model, feature_dim

    def extract_structure_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.structure_backbone_type == "swin":
            feat = self.structure_backbone.features(x)
            feat = self.structure_backbone.norm(feat)
            feat = self.structure_backbone.permute(feat)
            feat = self.structure_backbone.avgpool(feat)
            return torch.flatten(feat, 1)

        if self.structure_backbone_type == "dinov3":
            feat = self.structure_backbone.forward_features(x)
            if isinstance(feat, dict):
                cls_feat = feat.get("x_norm_clstoken", None)
                if torch.is_tensor(cls_feat):
                    return cls_feat

                patch_feat = feat.get("x_norm_patchtokens", None)
                if torch.is_tensor(patch_feat):
                    return patch_feat.mean(dim=1)

                last_hidden_state = feat.get("last_hidden_state", None)
                if torch.is_tensor(last_hidden_state):
                    return last_hidden_state[:, 0, :]

                for value in feat.values():
                    if torch.is_tensor(value):
                        if value.dim() == 3:
                            return value[:, 0, :]
                        if value.dim() == 2:
                            return value
                raise RuntimeError("DINOv3 forward_features returned dict without usable tensor features")

            if torch.is_tensor(feat):
                if feat.dim() == 3:
                    return feat[:, 0, :]
                if feat.dim() == 2:
                    return feat
            raise RuntimeError("DINOv3 forward_features returned unsupported feature format")

        raise RuntimeError(f"Unsupported structure backbone type: {self.structure_backbone_type}")

    def extract_swin_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_structure_features(x)

    def extract_clip_features(self, x: torch.Tensor) -> torch.Tensor:
        clip_image_size = getattr(getattr(self.clip_backbone, "config", None), "image_size", None)
        if isinstance(clip_image_size, int) and x.shape[-1] != clip_image_size:
            x = F.interpolate(x, size=(clip_image_size, clip_image_size), mode="bilinear", align_corners=False)
        try:
            outputs = self.clip_backbone(
                pixel_values=x,
                interpolate_pos_encoding=self.clip_interpolate_pos_encoding,
            )
        except TypeError:
            outputs = self.clip_backbone(pixel_values=x)
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        ref_img: torch.Tensor,
        dist_img: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_structure_ref = self.extract_structure_features(ref_img)
        feat_structure_dist = self.extract_structure_features(dist_img)
        feat_clip_ref = self.extract_clip_features(ref_img)
        feat_clip_dist = self.extract_clip_features(dist_img)

        if self.ablation_mode == "clip_only":
            feat_structure_ref = torch.zeros_like(feat_structure_ref)
            feat_structure_dist = torch.zeros_like(feat_structure_dist)
        elif self.ablation_mode == "structure_only":
            feat_clip_ref = torch.zeros_like(feat_clip_ref)
            feat_clip_dist = torch.zeros_like(feat_clip_dist)

        diff_structure = torch.abs(feat_structure_ref - feat_structure_dist)
        diff_clip = torch.abs(feat_clip_ref - feat_clip_dist)

        mult_clip = feat_clip_ref * feat_clip_dist
        if self.clip_mult_enabled and self.clip_mult_l2_norm:
            mult_clip = F.normalize(feat_clip_ref, dim=1) * F.normalize(feat_clip_dist, dim=1)

        cos_sim = F.cosine_similarity(feat_clip_ref, feat_clip_dist, dim=1).unsqueeze(1)
        if self.clip_mult_enabled and self.clip_mult_replace_raw:
            clip_fused_parts = [feat_clip_ref, diff_clip, mult_clip]
        elif self.clip_mult_enabled:
            clip_fused_parts = [feat_clip_ref, feat_clip_dist, diff_clip, mult_clip]
        else:
            clip_fused_parts = [feat_clip_ref, feat_clip_dist, diff_clip]

        fused = torch.cat(
            [
                feat_structure_ref,
                feat_structure_dist,
                diff_structure,
                *clip_fused_parts,
                cos_sim,
            ],
            dim=1,
        )

        fused = self.bottleneck(fused)
        logits = self.head(fused)

        cos_flat = cos_sim.squeeze(1)
        hard_gate_mask = torch.zeros_like(cos_flat, dtype=torch.bool)
        soft_gate_mask = torch.zeros_like(cos_flat, dtype=torch.bool)

        apply_semantic_gate = self.semantic_gate_enabled and (self.ablation_mode != "structure_only")
        if apply_semantic_gate and (not self.training):
            mode = self.semantic_gate_mode
            if mode == "off":
                mode = "off"
            elif mode not in {"hard", "soft"}:
                mode = "hard"

            low_th = self.semantic_gate_threshold
            high_th = max(self.semantic_gate_high_threshold, low_th + 1e-6)

            if mode == "hard":
                hard_gate_mask = cos_flat < low_th
                if torch.any(hard_gate_mask):
                    logits = logits.clone()
                    logits[hard_gate_mask] = -self.gate_logit_strength
                    logits[hard_gate_mask, 0] = self.gate_logit_strength

            if mode == "soft":
                hard_gate_mask = cos_flat < low_th
                soft_gate_mask = (cos_flat >= low_th) & (cos_flat < high_th)

                if torch.any(hard_gate_mask) or torch.any(soft_gate_mask):
                    logits = logits.clone()

                    if torch.any(hard_gate_mask):
                        logits[hard_gate_mask] = -self.gate_logit_strength
                        logits[hard_gate_mask, 0] = self.gate_logit_strength

                    if torch.any(soft_gate_mask):
                        alpha = (high_th - cos_flat[soft_gate_mask]) / (high_th - low_th)
                        alpha = alpha.clamp(min=0.0, max=1.0)
                        delta = alpha * self.soft_gate_logit_strength
                        logits[soft_gate_mask, 0] += delta
                        logits[soft_gate_mask, 1:] -= delta.unsqueeze(1)

        gate_mask = hard_gate_mask | soft_gate_mask

        probs = F.softmax(logits, dim=1)
        expected_scores = torch.sum(probs * self.score_weights, dim=1)
        if return_aux:
            return logits, expected_scores, {
                "cos_sim": cos_sim.squeeze(1),
                "gate_mask": gate_mask,
                "hard_gate_mask": hard_gate_mask,
                "soft_gate_mask": soft_gate_mask,
            }
        return logits, expected_scores
