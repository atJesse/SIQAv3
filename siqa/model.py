from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseSemanticIQA(nn.Module):
    def __init__(self, num_classes: int = 6, backbone_name: str = "dinov2_vits14", freeze_backbone: bool = True):
        super().__init__()

        self.backbone_name = backbone_name
        self.backbone, self.feature_dim = self._build_backbone(backbone_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        fusion_dim = self.feature_dim * 3
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, num_classes),
        )
        self.register_buffer("score_weights", torch.arange(0, num_classes, dtype=torch.float32))

    def _build_backbone(self, backbone_name: str):
        if backbone_name.startswith("dinov2"):
            model = torch.hub.load("facebookresearch/dinov2", backbone_name)
            feature_dim = 384 if "vits" in backbone_name else 768
            return model, feature_dim

        raise ValueError(f"Unsupported backbone: {backbone_name}")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.ndim > 2:
            feat = feat.flatten(1)
        return feat

    def forward(self, ref_img: torch.Tensor, dist_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_ref = self.extract_features(ref_img)
        feat_dist = self.extract_features(dist_img)
        feat_diff = torch.abs(feat_ref - feat_dist)
        fused = torch.cat([feat_ref, feat_dist, feat_diff], dim=1)

        logits = self.head(fused)
        probs = F.softmax(logits, dim=1)
        expected_scores = torch.sum(probs * self.score_weights, dim=1)
        return logits, expected_scores
