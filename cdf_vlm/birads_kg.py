"""BI-RADS Knowledge Graph for mammography text prompt enhancement.

Encodes clinical prior knowledge about lesion-malignancy associations from the
BI-RADS lexicon as a differentiable knowledge graph. Used as auxiliary
regularization: the text prompt encoder's learned keyword embeddings should
respect known malignancy associations.

Key relationships (from BI-RADS 5th edition):
  - Spiculated margin → strongly associated with malignancy
  - Circumscribed margin → typically benign (unless other suspicious features)
  - Pleomorphic/linear calcifications → high suspicion
  - Clustered distribution → moderate suspicion
  - Amorphous calcifications → low suspicion
  - Irregular shape → suspicious
  - Oval/round shape with circumscribed margin → typically benign
  - Architectural distortion → suspicious
  - Breast density 3-4 → reduced sensitivity, not a direct malignancy indicator
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# ══════════════════════════════════════════════════════════════
# BI-RADS Malignancy Prior Knowledge
# ══════════════════════════════════════════════════════════════

# Malignancy association scores for each of the 16 keywords
# +1.0 = strong malignancy indicator, -1.0 = strong benign indicator, 0 = neutral
MALIGNANCY_PRIORS = {
    # Lesion type
    "mass": 0.15,                    # Slightly more likely malignant overall
    "calcification": -0.10,          # More benign calcifications in dataset

    # Margin characteristics (mass)
    "spiculated": 0.85,              # Classic malignancy sign
    "circumscribed": -0.70,          # Typically benign (well-defined border)
    "obscured": -0.20,               # Non-specific, often benign
    "microlobulated": 0.55,          # Suspicious, often malignant
    "irregular": 0.60,               # Irregular shape → suspicious

    # Calcification morphology
    "pleomorphic": 0.75,             # Pleomorphic = varying sizes/shapes → malignant
    "linear": 0.80,                  # Linear/branching → highly suspicious (ductal)

    # Distribution
    "clustered": 0.25,               # Clustered → moderate suspicion
    "segmental": 0.50,               # Segmental → suspicious (ductal distribution)

    # Architectural
    "architectural distortion": 0.65, # Distortion → suspicious

    # Laterality / view (neutral — no diagnostic value)
    "left breast": 0.0,
    "right breast": 0.0,
    "cc view": 0.0,
    "mlo view": 0.0,
}

# Pairwise relationship strengths (cosine similarity target in embedding space)
# Positive = embeddings should be close, Negative = embeddings should be apart
# Format: (keyword_a, keyword_b, target_similarity)
PAIRWISE_RELATIONSHIPS = [
    # Co-occurring malignancy features → close in embedding space
    ("spiculated", "irregular", 0.7),
    ("spiculated", "malignant", 0.8),   # implicit "malignant" concept
    ("pleomorphic", "linear", 0.6),
    ("pleomorphic", "malignant", 0.7),
    ("linear", "malignant", 0.7),
    ("architectural distortion", "spiculated", 0.5),

    # Benign features → close to each other
    ("circumscribed", "benign", 0.7),   # implicit "benign" concept
    ("obscured", "benign", 0.3),

    # Discordant pairs → far apart in embedding space
    ("spiculated", "circumscribed", -0.5),
    ("pleomorphic", "circumscribed", -0.4),
    ("linear", "circumscribed", -0.4),

    # Anatomical neutrals → orthonormal to diagnostic features
    ("left breast", "spiculated", 0.0),
    ("right breast", "spiculated", 0.0),
    ("cc view", "pleomorphic", 0.0),
    ("mlo view", "pleomorphic", 0.0),
]


# ══════════════════════════════════════════════════════════════
# Knowledge graph regularization loss
# ══════════════════════════════════════════════════════════════

class BIRADSKnowledgeGraphLoss(nn.Module):
    """Auxiliary loss that encourages keyword embeddings to respect BI-RADS priors.

    Two components:
      1. Malignancy prior loss: each keyword's learned embedding direction
         should correlate with its known malignancy association.
      2. Pairwise relationship loss: known co-occurring features should be
         close in embedding space; discordant features should be far apart.
    """

    def __init__(
        self,
        keywords: list[str],
        priors: dict[str, float] | None = None,
        relationships: list[tuple[str, str, float]] | None = None,
        lambda_prior: float = 0.01,
        lambda_pairwise: float = 0.005,
    ):
        super().__init__()
        self.keywords = keywords
        self.priors = priors or MALIGNANCY_PRIORS
        self.relationships = relationships or PAIRWISE_RELATIONSHIPS
        self.lambda_prior = lambda_prior
        self.lambda_pairwise = lambda_pairwise

        # Build keyword → index mapping
        self.kw2idx = {kw: i for i, kw in enumerate(keywords)}

        # Malignancy prior targets [num_keywords]
        prior_values = [self.priors.get(kw, 0.0) for kw in keywords]
        self.register_buffer("prior_targets", torch.tensor(prior_values))

    def forward(self, keyword_embeddings: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute knowledge graph regularization loss.

        Args:
            keyword_embeddings: [num_keywords, embed_dim] — each row is the
                                learned embedding for one keyword.

        Returns:
            (total_loss, metrics_dict)
        """
        loss = keyword_embeddings.new_tensor(0.0)
        metrics = {}

        # 1) Malignancy prior loss: correlation between embedding norm and prior
        norms = keyword_embeddings.norm(dim=-1)  # [K]
        # Normalize norms and priors to [0,1] for correlation
        norms_norm = (norms - norms.mean()) / (norms.std() + 1e-8)
        priors_norm = self.prior_targets
        # MSE between normalized norm and prior target
        prior_loss = F.mse_loss(norms_norm, priors_norm)
        loss = loss + self.lambda_prior * prior_loss
        metrics["kg_prior_loss"] = float(prior_loss.detach())

        # 2) Pairwise relationship loss
        if len(self.relationships) > 0:
            pairwise_loss = keyword_embeddings.new_tensor(0.0)
            count = 0
            for kw_a, kw_b, target_sim in self.relationships:
                if kw_a in self.kw2idx and kw_b in self.kw2idx:
                    idx_a = self.kw2idx[kw_a]
                    idx_b = self.kw2idx[kw_b]
                    emb_a = F.normalize(keyword_embeddings[idx_a], dim=-1)
                    emb_b = F.normalize(keyword_embeddings[idx_b], dim=-1)
                    actual_sim = (emb_a * emb_b).sum()
                    pairwise_loss = pairwise_loss + F.mse_loss(actual_sim,
                        keyword_embeddings.new_tensor(target_sim))
                    count += 1
            if count > 0:
                pairwise_loss = pairwise_loss / count
                loss = loss + self.lambda_pairwise * pairwise_loss
                metrics["kg_pairwise_loss"] = float(pairwise_loss.detach())

        metrics["kg_total_loss"] = float(loss.detach())
        return loss, metrics


# ══════════════════════════════════════════════════════════════
# Knowledge-Graph-Enhanced Prompt Encoder
# ══════════════════════════════════════════════════════════════

class KGEnhancedPromptEncoder(nn.Module):
    """Text prompt encoder with BI-RADS knowledge graph regularization.

    Wraps PathologyPromptEncoder and adds:
      - Learnable keyword embeddings (one per keyword) as the first layer
        (replaces the first Linear layer of the original encoder).
      - BI-RADS knowledge graph loss as auxiliary regularization.
      - The KG loss is only active during training; at inference, the encoder
        works identically to the original.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        keywords: list[str] | None = None,
        kg_lambda_prior: float = 0.01,
        kg_lambda_pairwise: float = 0.005,
    ):
        super().__init__()
        from cdf_vlm.text_prompts import DEFAULT_KEYWORDS
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.num_keywords = len(self.keywords)

        # Learnable per-keyword embeddings (replaces first Linear layer)
        self.keyword_embeddings = nn.Parameter(
            torch.randn(self.num_keywords, embed_dim) * 0.02
        )

        # Projection: weighted sum of keyword embeddings → prompt vector
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Knowledge graph loss
        self.kg_loss_fn = BIRADSKnowledgeGraphLoss(
            keywords=self.keywords,
            priors=MALIGNANCY_PRIORS,
            relationships=PAIRWISE_RELATIONSHIPS,
            lambda_prior=kg_lambda_prior,
            lambda_pairwise=kg_lambda_pairwise,
        )

    def encode_texts(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        """Encode text descriptions into prompt vectors via keyword embedding weighted sum."""
        from cdf_vlm.text_prompts import keyword_multihot
        if device is None:
            device = self.keyword_embeddings.device
        multi_hot = keyword_multihot(texts, self.keywords).to(device)  # [B, K]
        # Normalize to avoid scale issues
        weights = multi_hot / (multi_hot.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K]
        combined = weights @ self.keyword_embeddings  # [B, D]
        return self.proj(combined)

    def forward(self, keyword_features: torch.Tensor) -> torch.Tensor:
        """Forward with multi-hot input (for compatibility with old interface)."""
        if keyword_features.dim() == 2 and keyword_features.shape[1] == self.num_keywords:
            weights = keyword_features / (keyword_features.sum(dim=-1, keepdim=True) + 1e-8)
            combined = weights @ self.keyword_embeddings
            return self.proj(combined)
        return self.proj(keyword_features)

    def compute_kg_loss(self) -> tuple[torch.Tensor, dict]:
        """Compute knowledge graph regularization loss for the current embeddings."""
        return self.kg_loss_fn(self.keyword_embeddings)


# ══════════════════════════════════════════════════════════════
# Analysis: keyword embedding similarity vs clinical priors
# ══════════════════════════════════════════════════════════════

def analyze_keyword_embeddings(
    encoder: KGEnhancedPromptEncoder,
) -> dict:
    """Analyze learned keyword embeddings for clinical consistency.

    Returns:
      - Embedding similarity matrix [K, K]
      - Malignancy correlation: Pearson r between embedding norm and prior
      - Pairwise agreement: fraction of relationship targets correctly ordered
    """
    import numpy as np

    embeddings = encoder.keyword_embeddings.detach()
    keywords = encoder.keywords
    K = len(keywords)

    # Similarity matrix
    emb_norm = F.normalize(embeddings, dim=-1)
    sim_matrix = (emb_norm @ emb_norm.T).cpu().numpy()  # [K, K]

    # Malignancy correlation
    norms = embeddings.norm(dim=-1).cpu().numpy()
    prior_vals = np.array([MALIGNANCY_PRIORS.get(kw, 0.0) for kw in keywords])
    # Pearson r
    r_num = np.sum((norms - norms.mean()) * (prior_vals - prior_vals.mean()))
    r_den = np.sqrt(np.sum((norms - norms.mean())**2) * np.sum((prior_vals - prior_vals.mean())**2))
    r = r_num / (r_den + 1e-8)

    # Pairwise agreement
    agreements = []
    for kw_a, kw_b, target_sim in PAIRWISE_RELATIONSHIPS:
        if kw_a in keywords and kw_b in keywords:
            i, j = keywords.index(kw_a), keywords.index(kw_b)
            actual = float(sim_matrix[i, j])
            agreements.append({
                "pair": f"{kw_a}↔{kw_b}",
                "target": target_sim,
                "actual": actual,
                "error": abs(actual - target_sim),
            })

    return {
        "keyword_norms": {kw: float(n) for kw, n in zip(keywords, norms)},
        "malignancy_pearson_r": float(r),
        "pairwise_agreement": sorted(agreements, key=lambda x: x["error"]),
        "sim_matrix": sim_matrix,
        "keywords": keywords,
    }


# ══════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    encoder = KGEnhancedPromptEncoder(embed_dim=512)

    # Test encoding
    texts = [
        "spiculated mass, irregular shape, right breast, CC view",
        "circumscribed mass, oval shape, left breast, MLO view",
        "pleomorphic calcification, linear distribution, left breast, CC view",
    ]
    vecs = encoder.encode_texts(texts)
    print(f"Encoded {len(texts)} descriptions → {vecs.shape}")

    # KG loss
    kg_loss, metrics = encoder.compute_kg_loss()
    print(f"KG loss: {kg_loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Analyze embeddings
    analysis = analyze_keyword_embeddings(encoder)
    print(f"\nMalignancy Pearson r: {analysis['malignancy_pearson_r']:.4f}")
    print("\nKeyword norms (top 5):")
    sorted_kw = sorted(analysis["keyword_norms"].items(), key=lambda x: x[1], reverse=True)
    for kw, norm in sorted_kw[:5]:
        prior = MALIGNANCY_PRIORS.get(kw, 0.0)
        print(f"  {kw:<30} norm={norm:.4f}  prior={prior:+.2f}")

    print("\nPairwise agreement (worst 3 errors):")
    for a in analysis["pairwise_agreement"][-3:]:
        print(f"  {a['pair']:<30} target={a['target']:+.1f} actual={a['actual']:+.3f} error={a['error']:.3f}")
