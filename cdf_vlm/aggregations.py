"""Alternative keyword aggregation encoders for the AAMAS rebuttal ablation.

Variants over the linear weighted-average aggregation (Eq. 1 of the paper):
  - "linear":    normalized weighted sum of keyword embeddings (paper default)
  - "attention": softmax attention pooling over the present keywords
  - "bilinear":  linear aggregate plus a low-rank pairwise (second-order)
                 interaction term, added back into the aggregate space

All variants share the same keyword embedding table and can optionally apply
the BI-RADS knowledge graph regularization (malignancy prior + pairwise
relation losses), exactly as KGEnhancedPromptEncoder does.
"""

from __future__ import annotations

import torch
from torch import nn

from cdf_vlm.birads_kg import BIRADSKnowledgeGraphLoss, MALIGNANCY_PRIORS, PAIRWISE_RELATIONSHIPS


class AggPromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        keywords: list[str] | None = None,
        aggregation: str = "linear",
        bilinear_dim: int = 32,
        kg_lambda_prior: float = 0.01,
        kg_lambda_pairwise: float = 0.005,
        use_kg: bool = True,
    ):
        super().__init__()
        from cdf_vlm.text_prompts import DEFAULT_KEYWORDS
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.num_keywords = len(self.keywords)
        self.aggregation = aggregation

        self.keyword_embeddings = nn.Parameter(
            torch.randn(self.num_keywords, embed_dim) * 0.02
        )

        if aggregation == "attention":
            self.attn_score = nn.Linear(embed_dim, 1)
        elif aggregation == "bilinear":
            self.inter_u = nn.Linear(embed_dim, bilinear_dim, bias=False)
            self.inter_v = nn.Linear(embed_dim, bilinear_dim, bias=False)
            self.inter_out = nn.Linear(bilinear_dim, embed_dim, bias=False)
        elif aggregation != "linear":
            raise ValueError(f"Unknown aggregation: {aggregation}")

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.kg_loss_fn = None
        if use_kg:
            self.kg_loss_fn = BIRADSKnowledgeGraphLoss(
                keywords=self.keywords,
                priors=MALIGNANCY_PRIORS,
                relationships=PAIRWISE_RELATIONSHIPS,
                lambda_prior=kg_lambda_prior,
                lambda_pairwise=kg_lambda_pairwise,
            )

    def _multihot(self, texts: list[str], device: torch.device) -> torch.Tensor:
        from cdf_vlm.text_prompts import keyword_multihot
        return keyword_multihot(texts, self.keywords).to(device)

    def encode_texts(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = self.keyword_embeddings.device
        multi_hot = self._multihot(texts, device)  # [B, K]
        weights = multi_hot / (multi_hot.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K]
        E = self.keyword_embeddings  # [K, D]

        if self.aggregation == "linear":
            combined = weights @ E  # [B, D]
        elif self.aggregation == "attention":
            scores = self.attn_score(E).squeeze(-1)  # [K]
            logits = scores.unsqueeze(0).masked_fill(multi_hot == 0, float("-inf"))
            attn = torch.softmax(logits, dim=-1)  # [B, K]
            combined = attn @ E  # [B, D]
        elif self.aggregation == "bilinear":
            linear_part = weights @ E  # [B, D]
            u = self.inter_u(E)  # [K, d]
            v = self.inter_v(E)  # [K, d]
            su = weights @ u  # [B, d]
            sv = weights @ v  # [B, d]
            self_term = (weights * weights) @ (u * v)  # [B, d]
            inter = su * sv - self_term  # sum over j != k of w_j w_k (u_j * v_k)
            denom = 1.0 - (weights * weights).sum(dim=-1, keepdim=True)  # [B, 1]
            inter = inter / (denom.clamp_min(1e-6) + 1e-8)
            combined = linear_part + self.inter_out(inter)  # [B, D]

        return self.proj(combined)

    def compute_kg_loss(self):
        if self.kg_loss_fn is None:
            raise RuntimeError("KG loss not enabled for this encoder")
        return self.kg_loss_fn(self.keyword_embeddings)
