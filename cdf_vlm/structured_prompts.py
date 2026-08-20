"""Structured-field text prompt encoder for CBIS-DDSM structured descriptions.

Motivation: The current PathologyPromptEncoder uses 16 binary keywords, which discards:
  - Calcification type (AMORPHOUS, PLEOMORPHIC, PUNCTATE, COARSE, etc.)
  - Distribution pattern (CLUSTERED, LINEAR, SEGMENTAL, REGIONAL, DIFFUSE)
  - Mass shape (IRREGULAR, LOBULATED, OVAL, ROUND, etc.)
  - Mass margin (SPICULATED, CIRCUMSCRIBED, OBSCURED, etc.)
  - Breast density (1-4)
  - BI-RADS assessment (0-5)
  - Subtlety rating (1-5)

This module parses the structured CBIS-DDSM description text into typed fields
and learns per-value embeddings, providing richer semantic features than binary
keyword presence/absence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn


# ══════════════════════════════════════════════════════════════
# Field vocabulary (populated from actual CBIS-DDSM data)
# ══════════════════════════════════════════════════════════════

CALCIFICATION_TYPES = [
    "amorphous", "pleomorphic", "punctate", "coarse",
    "fine_linear_branching", "rod_like", "round", "dystrophic",
    "lucent_centered", "milk_of_calcium", "skin", "vascular", "large",
    "egg_shell", "rim", "popcorn", "scattered", "fat necrosis",
]

DISTRIBUTION_TYPES = [
    "clustered", "linear", "segmental", "regional", "diffuse", "grouped",
]

MASS_SHAPES = [
    "irregular", "lobulated", "oval", "round", "architectural_distortion",
    "focal_asymmetry", "asymmetry",
]

MASS_MARGINS = [
    "spiculated", "circumscribed", "obscured", "microlobulated",
    "indistinct", "ill_defined",
]

ABNORMALITY_TYPES = ["calcification", "mass"]

BREAST_DENSITY_LEVELS = [1, 2, 3, 4]

ASSESSMENT_LEVELS = [0, 1, 2, 3, 4, 5]

SUBTLEty_LEVELS = [1, 2, 3, 4, 5]


@dataclass
class StructuredFields:
    """Parsed structured fields from a CBIS-DDSM description string."""

    abnormality_type: str = "unknown"       # calcification or mass
    calcification_type: str = "unknown"     # AMORPHOUS, PLEOMORPHIC, etc.
    distribution: str = "unknown"           # CLUSTERED, LINEAR, etc.
    mass_shape: str = "unknown"             # IRREGULAR, LOBULATED, etc.
    mass_margin: str = "unknown"            # SPICULATED, CIRCUMSCRIBED, etc.
    breast_density: float = 0.0             # 1-4
    assessment: int = 0                     # 0-5
    subtlety: int = 0                       # 1-5
    laterality: str = "unknown"             # left or right
    view: str = "unknown"                   # CC or MLO

    # Additional keyword presence (same as old 16-keyword for backward compat)
    keywords_present: list[str] = field(default_factory=list)


def parse_structured_description(text: str) -> StructuredFields:
    """Parse a CBIS-DDSM structured description into typed fields.

    Example input:
      "breast density is 3.0, right breast, CC view, abnormality id 1,
       abnormality type calcification, PLEOMORPHIC type, LINEAR distribution,
       assessment 4, subtlety 4"

    Returns StructuredFields with typed values.
    """
    text_l = text.lower().strip()
    fields = StructuredFields()

    # Abnormality type
    type_match = re.search(r'abnormality type\s+(\w+)', text_l)
    if type_match:
        fields.abnormality_type = type_match.group(1)

    # Calcification type: matches the word before "type" that follows "calcification"
    # e.g. "abnormality type calcification, AMORPHOUS type" → AMORPHOUS
    # e.g. "abnormality type calcification, PLEOMORPHIC type" → PLEOMORPHIC
    calc_match = re.search(r'calcification,\s*(\w+)\s+type', text_l)
    if calc_match:
        ct = calc_match.group(1).lower()
        if ct not in ("nan", "unknown"):
            fields.calcification_type = ct
    # Fallback: look for any word+type pattern not preceded by "abnormality"
    if fields.calcification_type == "unknown":
        for m in re.finditer(r'(?<!abnormality\s)type\s+(\w+)', text_l):
            val = m.group(1).lower()
            if val != "nan":
                fields.calcification_type = val
                break

    # Distribution
    dist_match = re.search(r'(\w+)\s+distribution', text_l)
    if dist_match:
        fields.distribution = dist_match.group(1).lower()

    # Mass shape: "IRREGULAR-ARCHITECTURAL_DISTORTION shape" or "OVAL shape"
    shape_match = re.search(r'([\w_-]+)\s+shape', text_l)
    if shape_match:
        raw = shape_match.group(1).lower().replace('_', '-')
        # Normalize compound shapes: take the first/primary descriptor
        if '-' in raw:
            raw = raw.split('-')[0]  # "irregular-architectural_distortion" → "irregular"
        fields.mass_shape = raw

    # Mass margin: "SPICULATED margins" or "ILL_DEFINED margins"
    margin_match = re.search(r'([\w_-]+)\s+margins?', text_l)
    if margin_match:
        raw = margin_match.group(1).lower().replace('_', '-')
        fields.mass_margin = raw

    # Breast density
    density_match = re.search(r'breast density is\s+([\d.]+)', text_l)
    if density_match:
        try:
            fields.breast_density = float(density_match.group(1))
        except ValueError:
            pass

    # Assessment
    assess_match = re.search(r'assessment\s+(\d+)', text_l)
    if assess_match:
        fields.assessment = int(assess_match.group(1))

    # Subtlety
    subtle_match = re.search(r'subtlety\s+(\d+)', text_l)
    if subtle_match:
        fields.subtlety = int(subtle_match.group(1))

    # Laterality
    if "left" in text_l:
        fields.laterality = "left"
    elif "right" in text_l:
        fields.laterality = "right"

    # View
    if "cc view" in text_l:
        fields.view = "cc"
    elif "mlo view" in text_l:
        fields.view = "mlo"

    # Extract legacy keywords for backward compatibility
    from cdf_vlm.text_prompts import DEFAULT_KEYWORDS, extract_pathology_keywords
    fields.keywords_present = extract_pathology_keywords(text, DEFAULT_KEYWORDS)

    return fields


# ══════════════════════════════════════════════════════════════
# Structured-field prompt encoder (replaces PathologyPromptEncoder)
# ══════════════════════════════════════════════════════════════

class StructuredFieldPromptEncoder(nn.Module):
    """Encodes CBIS-DDSM structured descriptions into rich prompt vectors.

    Learns embeddings for:
      - Calcification types (18 values)
      - Distribution types (6 values)
      - Mass shapes (7 values)
      - Mass margins (7 values)
      - Abnormality type (2 values)
      - Laterality (2 values)
      - View (2 values)

    Plus continuous features (density, assessment, subtlety) normalized to [0,1].

    All embeddings are concatenated and projected through a 2-layer MLP
    to produce a single 512-dim prompt vector.

    Total vocabulary: ~44 categorical values + 3 continuous → richer than 16 binary bits.
    """

    def __init__(self, embed_dim: int = 512, per_field_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.per_field_dim = per_field_dim

        # Categorical embeddings
        self.calc_type_emb = nn.Embedding(len(CALCIFICATION_TYPES) + 1, per_field_dim, padding_idx=0)
        self.dist_emb = nn.Embedding(len(DISTRIBUTION_TYPES) + 1, per_field_dim, padding_idx=0)
        self.shape_emb = nn.Embedding(len(MASS_SHAPES) + 1, per_field_dim, padding_idx=0)
        self.margin_emb = nn.Embedding(len(MASS_MARGINS) + 1, per_field_dim, padding_idx=0)
        self.abnorm_emb = nn.Embedding(len(ABNORMALITY_TYPES) + 1, per_field_dim, padding_idx=0)
        self.laterality_emb = nn.Embedding(3, per_field_dim, padding_idx=0)  # unknown, left, right
        self.view_emb = nn.Embedding(3, per_field_dim, padding_idx=0)        # unknown, cc, mlo

        # Legacy keyword multi-hot (16-dim, for backward compatibility)
        self.keyword_bits = 16

        # Total input dim: 7 categorical × per_field_dim + 3 continuous + 16 keyword_bits
        total_in = 7 * per_field_dim + 3 + self.keyword_bits
        self.proj = nn.Sequential(
            nn.Linear(total_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Field name → index lookup
        self._calc_idx = {name: i + 1 for i, name in enumerate(CALCIFICATION_TYPES)}
        self._dist_idx = {name: i + 1 for i, name in enumerate(DISTRIBUTION_TYPES)}
        self._shape_idx = {name: i + 1 for i, name in enumerate(MASS_SHAPES)}
        self._margin_idx = {name: i + 1 for i, name in enumerate(MASS_MARGINS)}
        self._abnorm_idx = {name: i + 1 for i, name in enumerate(ABNORMALITY_TYPES)}
        self._lat_idx = {"unknown": 0, "left": 1, "right": 2}
        self._view_idx = {"unknown": 0, "cc": 1, "mlo": 2}

    def _safe_idx(self, lookup: dict, key: str) -> int:
        return lookup.get(key.lower(), 0)

    def encode_texts(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        """Parse structured descriptions and encode to prompt vectors."""
        batch_fields = [parse_structured_description(t) for t in texts]
        return self.forward(batch_fields, device=device)

    def forward(self, fields_list: list[StructuredFields],
                device: torch.device | None = None) -> torch.Tensor:
        """Encode a batch of StructuredFields into prompt vectors.

        Returns: [batch_size, embed_dim]
        """
        if device is None:
            device = next(self.parameters()).device

        batch_size = len(fields_list)
        parts = []

        # Categorical embeddings
        calc_idx = torch.tensor(
            [self._safe_idx(self._calc_idx, f.calcification_type) for f in fields_list],
            device=device)
        parts.append(self.calc_type_emb(calc_idx))

        dist_idx = torch.tensor(
            [self._safe_idx(self._dist_idx, f.distribution) for f in fields_list],
            device=device)
        parts.append(self.dist_emb(dist_idx))

        shape_idx = torch.tensor(
            [self._safe_idx(self._shape_idx, f.mass_shape) for f in fields_list],
            device=device)
        parts.append(self.shape_emb(shape_idx))

        margin_idx = torch.tensor(
            [self._safe_idx(self._margin_idx, f.mass_margin) for f in fields_list],
            device=device)
        parts.append(self.margin_emb(margin_idx))

        abnorm_idx = torch.tensor(
            [self._safe_idx(self._abnorm_idx, f.abnormality_type) for f in fields_list],
            device=device)
        parts.append(self.abnorm_emb(abnorm_idx))

        lat_idx = torch.tensor(
            [self._lat_idx.get(f.laterality, 0) for f in fields_list], device=device)
        parts.append(self.laterality_emb(lat_idx))

        view_idx = torch.tensor(
            [self._view_idx.get(f.view, 0) for f in fields_list], device=device)
        parts.append(self.view_emb(view_idx))

        # Continuous features (normalized to [0, 1])
        density = torch.tensor(
            [[f.breast_density / 4.0] for f in fields_list], device=device)
        assessment = torch.tensor(
            [[f.assessment / 5.0] for f in fields_list], device=device)
        subtlety = torch.tensor(
            [[f.subtlety / 5.0] for f in fields_list], device=device)
        parts.append(density)
        parts.append(assessment)
        parts.append(subtlety)

        # Legacy keyword multi-hot
        from cdf_vlm.text_prompts import keyword_multihot
        keyword_vecs = keyword_multihot(
            [", ".join(f.keywords_present) for f in fields_list]
        ).to(device)
        parts.append(keyword_vecs)

        # Concatenate and project
        combined = torch.cat(parts, dim=-1)  # [B, total_in]
        return self.proj(combined)


# ══════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    samples = [
        "breast density is 3.0, right breast, CC view, abnormality id 1, "
        "abnormality type calcification, AMORPHOUS type, CLUSTERED distribution, "
        "assessment 3, subtlety 3",
        "breast density is 4.0, left breast, MLO view, abnormality id 1, "
        "abnormality type calcification, PLEOMORPHIC type, LINEAR distribution, "
        "assessment 4, subtlety 4",
        "breast density is 2.0, right breast, CC view, abnormality id 1, "
        "abnormality type mass, mass shape irregular, margin spiculated, "
        "assessment 5, subtlety 5",
    ]

    encoder = StructuredFieldPromptEncoder(embed_dim=512, per_field_dim=32)
    vectors = encoder.encode_texts(samples)
    print(f"Encoded {len(samples)} descriptions → {vectors.shape}")
    print(f"  Old approach: 16-dim multi-hot → 512-dim (2-layer MLP)")
    print(f"  New approach: {7*32+3+16}-dim structured → 512-dim (2-layer MLP)")

    # Show parsed fields for verification
    for i, s in enumerate(samples):
        f = parse_structured_description(s)
        print(f"\n  [{i}] {s[:80]}...")
        print(f"       calc_type={f.calcification_type}, dist={f.distribution}, "
              f"shape={f.mass_shape}, margin={f.mass_margin}")
        print(f"       density={f.breast_density}, assess={f.assessment}, "
              f"subtlety={f.subtlety}, lat={f.laterality}, view={f.view}")
        print(f"       keywords={f.keywords_present[:8]}")
