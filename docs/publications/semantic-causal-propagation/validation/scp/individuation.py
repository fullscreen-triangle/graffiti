"""Individuation by negation and the recognition/search identity.

Implements Theorem 4.2 (Individuation by Negation), Corollary 4.3, and
Theorem 4.5 (Recognition/Search Identity) of
``semantic-causal-propagation.tex``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set


def complement(whole: FrozenSet[str], subset: FrozenSet[str]) -> FrozenSet[str]:
    """co(U) = V \\ U."""
    return whole - subset


def double_complement_is_identity(whole: FrozenSet[str], subset: FrozenSet[str]) -> bool:
    """Verify co(co(U)) == U (Theorem 4.2 involution)."""
    return complement(whole, complement(whole, subset)) == subset


@dataclass
class Decoder:
    """A finite decoder Dec: Q -> V and its fibre map Proj (Definition 4.4).

    ``mapping`` gives Dec directly; Proj(v) is derived as the fibre
    {q : Dec(q) = v}, exactly as Theorem 4.5 requires.
    """

    mapping: Dict[str, str]  # query -> claim

    def decode(self, query: str) -> str:
        return self.mapping[query]

    def fibre(self, claim: str) -> Set[str]:
        return {q for q, v in self.mapping.items() if v == claim}

    def recovers_decoder(self) -> Dict[str, str]:
        """Reconstruct Dec from the fibre partition {Proj(v)}_v, per Theorem 4.5 proof."""
        claims = set(self.mapping.values())
        rebuilt: Dict[str, str] = {}
        for v in claims:
            for q in self.fibre(v):
                rebuilt[q] = v
        return rebuilt


def build_synthetic_decoder(rng: random.Random, n_claims: int, n_queries_per_claim: int) -> Decoder:
    mapping: Dict[str, str] = {}
    for c in range(n_claims):
        claim = f"claim{c}"
        for q in range(n_queries_per_claim):
            mapping[f"q{c}_{q}"] = claim
    # Shuffle iteration order for good measure (dict insertion order aside).
    items = list(mapping.items())
    rng.shuffle(items)
    return Decoder(mapping=dict(items))
