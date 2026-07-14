"""
retriever/query_parser.py
---------------------------
Parses a free-text query into the SAME structured slot format produced by
indexer/attribute_extractor.py, so we can compare like-for-like at search
time instead of relying only on whole-query-vs-whole-image embedding
similarity (which is exactly where vanilla CLIP loses compositional
information).

This is deliberately a light rule-based parser over the shared vocabulary
in config.py, not a trained model -- for a small, well-defined slot schema
(color, garment, region, location, style) a vocabulary lookup is more
reliable and more debuggable than a generative parser, and it costs zero
extra model calls at query time. If the vocabulary needs to grow
significantly (e.g. multilingual queries, many more garment types), this
is the natural place to swap in a small LLM-based slot-filler -- the output
contract (the dict shape below) would stay identical, so nothing else in
retriever/search.py would need to change.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional

import config


def _find_color_garment_pairs(text: str) -> List[Dict]:
    """Find every (color, garment) pair mentioned in the query, in the
    order they appear, using a small window so 'a red tie and a white
    shirt' correctly yields [(red, tie), (white, shirt)] rather than
    cross-binding red-with-shirt."""
    text_l = text.lower()
    pairs = []
    # crude but effective: look for "<color> <0-2 words> <garment>" windows
    tokens = re.findall(r"[a-z']+", text_l)
    for i, tok in enumerate(tokens):
        if tok in config.COLORS:
            # search forward up to 3 tokens for a garment word/phrase
            window = " ".join(tokens[i + 1: i + 4])
            for garment in sorted(config.ALL_GARMENTS, key=len, reverse=True):
                if garment in window or garment.replace(" ", "") in window.replace(" ", ""):
                    region = _region_for_garment(garment)
                    pairs.append({"color": tok, "garment": garment, "region": region})
                    break
    return pairs


def _region_for_garment(garment: str) -> str:
    for region, garments in config.GARMENTS_BY_REGION.items():
        if garment in garments:
            return region
    return "full_body"


def _find_location(text: str) -> Optional[str]:
    text_l = text.lower()
    for loc_key, keywords in config.LOCATION_KEYWORDS.items():
        if any(kw in text_l for kw in keywords):
            return loc_key
    return None


def _find_style(text: str) -> Optional[str]:
    text_l = text.lower()
    for style_key, keywords in config.STYLE_KEYWORDS.items():
        if any(kw in text_l for kw in keywords):
            return style_key
    return None


def parse_query(text: str) -> Dict:
    """Returns:
    {
      "raw": original text,
      "color_garment_pairs": [{"color": "red", "garment": "tie", "region": "upper_body"}, ...],
      "location": "office" | None,
      "style": "formal" | None,
    }
    """
    return {
        "raw": text,
        "color_garment_pairs": _find_color_garment_pairs(text),
        "location": _find_location(text),
        "style": _find_style(text),
    }
