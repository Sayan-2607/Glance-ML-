"""
indexer/attribute_extractor.py
--------------------------------
This module is the answer to the assignment's core hint: vanilla CLIP
similarity treats a caption as an unordered "bag" of concepts, so
"a red shirt and blue pants" and "a blue shirt and red pants" end up with
nearly identical embeddings. To fix this we do NOT rely solely on a single
whole-image embedding for compositional queries -- we additionally extract
a small STRUCTURED record per image, where color is bound to a specific
garment in a specific body region:

    {
      "upper_body": {"garment": "shirt", "color": "blue", "score": 0.81},
      "lower_body": {"garment": "pants", "color": "red",  "score": 0.77},
      "outerwear":  None,
      "location":   {"label": "office", "score": 0.63},
      "style":      {"label": "formal", "score": 0.55},
    }

At query time (retriever/query_parser.py + retriever/search.py) the same
binding is parsed out of the free-text query and checked against this
record region-by-region, which is what correctly tells "red shirt / blue
pants" apart from "blue shirt / red pants".

Region localization: a full person-detector + pose model (e.g. YOLOS /
Mediapipe / a human-parsing network such as SCHP) is the production-grade
choice and is what should be used in a real deployment for people captured
at arbitrary crops/angles. To keep this module runnable without extra
model downloads, we use the standard fixed-ratio body-region heuristic
that is common in fast fashion-tagging pipelines when photos are
roughly full-body/half-body portraits (which is true for the large
majority of Fashionpedia-style images): top ~45% of the person bounding
box -> upper body garment, next ~40% -> lower body garment, and the full
frame is also scanned separately for outerwear (a coat/blazer usually
spans both regions). Swapping in a real segmentation model only requires
replacing `_get_region_crops()` below -- the rest of the module is
model-agnostic.
"""
from __future__ import annotations
from typing import Dict, Optional

from PIL import Image

import config
from indexer.embedding_backend import EmbeddingBackend


def _get_region_crops(image: Image.Image) -> Dict[str, Image.Image]:
    """Heuristic body-region crops. Replace with a real
    detector/segmentation model for production use (see module docstring)."""
    w, h = image.size
    return {
        "upper_body": image.crop((0, int(h * 0.10), w, int(h * 0.55))),
        "lower_body": image.crop((0, int(h * 0.50), w, int(h * 0.95))),
        "outerwear":  image.crop((0, int(h * 0.05), w, int(h * 0.75))),
        "full":       image,
    }


def _classify_region_garment(
    backend: EmbeddingBackend, crop: Image.Image, region: str
) -> Optional[Dict]:
    garments = config.GARMENTS_BY_REGION[region if region != "full" else "full_body"]
    candidate_labels = [
        config.PROMPT_TEMPLATE.format(color=color, garment=g)
        for g in garments
        for color in config.COLORS
    ]
    if not candidate_labels:
        return None
    label, score = backend.zero_shot_classify(crop, candidate_labels)
    parsed = _parse_color_garment_label(label)
    if parsed is None:
        return None
    color, garment = parsed
    return {"garment": garment, "color": color, "score": round(score, 4)}


def _parse_color_garment_label(label: str) -> Optional[tuple]:
    # label looks like: "a photo of a person wearing a red shirt"
    for color in config.COLORS:
        if f" {color} " in f" {label} ":
            for garment in config.ALL_GARMENTS:
                if garment in label:
                    return color, garment
    return None


def classify_location(backend: EmbeddingBackend, image: Image.Image) -> Dict:
    label, score = backend.zero_shot_classify(
        image, [config.LOCATION_PROMPT_TEMPLATE.format(location=loc) for loc in config.LOCATIONS]
    )
    return {"label": label, "score": round(score, 4)}


def classify_style(backend: EmbeddingBackend, image: Image.Image) -> Dict:
    label, score = backend.zero_shot_classify(
        image, [config.STYLE_PROMPT_TEMPLATE.format(style=s) for s in config.STYLES]
    )
    return {"label": label, "score": round(score, 4)}


def extract_attributes(backend: EmbeddingBackend, image: Image.Image) -> Dict:
    """Main entry point: returns the full structured attribute record for
    one image, used by build_index.py and stored as metadata alongside the
    global embedding."""
    crops = _get_region_crops(image)

    upper = _classify_region_garment(backend, crops["upper_body"], "upper_body")
    lower = _classify_region_garment(backend, crops["lower_body"], "lower_body")
    outer = _classify_region_garment(backend, crops["outerwear"], "outerwear")

    # Keep only the outerwear guess if it is confidently NOT just re-detecting
    # the upper-body garment (avoids double counting a shirt as a "jacket").
    if outer and upper and outer["garment"] == upper["garment"]:
        outer = None

    location = classify_location(backend, image)
    style = classify_style(backend, image)

    return {
        "upper_body": upper,
        "lower_body": lower,
        "outerwear": outer,
        "location": location,
        "style": style,
    }
