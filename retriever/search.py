"""
retriever/search.py
---------------------
PART B: THE RETRIEVER

Usage:
    python -m retriever.search --index outputs/index --backend offline \
        --query "A red tie and a white shirt in a formal setting" --k 5

Search strategy ("Structured Hybrid Retrieval"):
  1. Recall stage: embed the query text with the SAME backend used to build
     the index, and pull the top `config.TOP_K_CANDIDATES` nearest images
     by cosine similarity on the global embedding. This stage is what gives
     broad, zero-shot semantic recall (style, general vibe, unseen phrasing).
  2. Re-rank stage: parse the query into structured slots (query_parser.py)
     and score each candidate against its stored structured attributes
     (attribute_extractor.py output) region-by-region. This stage is what
     fixes compositionality -- "red tie" must match the *tie* region's
     color, not just correlate with "red" appearing somewhere in the image.
  3. Final score = weighted sum of (a) global embedding similarity,
     (b) fraction of requested (color, garment) pairs correctly bound,
     (c) location match, (d) style match. Weights are tunable via CLI flags
     and documented in the write-up.

This two-stage design (broad ANN recall -> structured re-rank) is a
standard, scalable pattern: stage 1 is O(log N) / vectorized regardless of
catalog size, and stage 2 only runs on a bounded candidate set (~100),
so cost does not grow with the size of the full catalog.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from indexer.embedding_backend import get_backend  # noqa: E402
from retriever.query_parser import parse_query  # noqa: E402
from vectorstore.store import FlatIndex, MetadataStore  # noqa: E402


def _attribute_match_score(parsed_query: Dict, attrs: Dict) -> Tuple[float, List[str]]:
    """Fraction of requested (color, garment) pairs found in the matching
    region of this image's stored attributes, plus location/style bonuses.
    Returns (score in [0,1], list of human-readable match explanations)."""
    explanations = []
    sub_scores = []

    pairs = parsed_query["color_garment_pairs"]
    if pairs:
        hits = 0
        for pair in pairs:
            region_key = pair["region"] if pair["region"] != "full_body" else "outerwear"
            region_attr = attrs.get(region_key) or attrs.get("outerwear")
            if region_attr and region_attr.get("color") == pair["color"]:
                hits += 1
                explanations.append(
                    f"matched {pair['color']} {pair['garment']} in {region_key}"
                )
        sub_scores.append(hits / len(pairs))

    if parsed_query["location"]:
        loc_label = (attrs.get("location") or {}).get("label", "")
        if parsed_query["location"] in loc_label.lower():
            sub_scores.append(1.0)
            explanations.append(f"location matched: {parsed_query['location']}")
        else:
            sub_scores.append(0.0)

    if parsed_query["style"]:
        style_label = (attrs.get("style") or {}).get("label", "")
        if parsed_query["style"] in style_label.lower():
            sub_scores.append(1.0)
            explanations.append(f"style matched: {parsed_query['style']}")
        else:
            sub_scores.append(0.0)

    if not sub_scores:
        return 0.0, explanations
    return sum(sub_scores) / len(sub_scores), explanations


def search(
    index_prefix: str,
    query: str,
    backend_name: str,
    k: int = 5,
    w_embed: float = 0.55,
    w_attrs: float = 0.45,
) -> List[Dict]:
    backend = get_backend(backend_name)
    index = FlatIndex.load(index_prefix)
    meta = MetadataStore.load(f"{index_prefix}.meta.json")

    query_vec = backend.encode_text(query)
    candidates = index.search(query_vec, k=config.TOP_K_CANDIDATES)

    parsed = parse_query(query)

    results = []
    for image_id, embed_score in candidates:
        record = meta.get(image_id)
        attrs = record["attributes"]
        attr_score, explanations = _attribute_match_score(parsed, attrs)
        final_score = w_embed * embed_score + w_attrs * attr_score
        results.append(
            {
                "image_id": image_id,
                "path": record["path"],
                "final_score": round(final_score, 4),
                "embed_score": round(embed_score, 4),
                "attr_score": round(attr_score, 4),
                "explanations": explanations,
            }
        )

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="index path prefix used in build_index.py --out")
    parser.add_argument("--query", required=True)
    parser.add_argument("--backend", default=config.DEFAULT_BACKEND, choices=["clip", "offline"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    hits = search(args.index, args.query, args.backend, args.k)
    print(json.dumps(hits, indent=2))
