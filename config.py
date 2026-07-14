"""
config.py
---------
Single source of truth for the controlled vocabularies used by BOTH the
indexer (Part A) and the retriever (Part B). Keeping this in one shared
module is what makes the ML logic "compositional": indexing time attribute
extraction and query time attribute parsing look up the *same* lists, so a
color/garment/location/style token means the same thing on both sides of
the pipeline.

Extend these lists to widen coverage -- nothing else in the codebase needs
to change for new vocabulary to start working (this is what "zero-shot"
means in practice here: we are not retraining a classifier, we are handing
CLIP a new prompt).
"""

# ---------------------------------------------------------------------------
# Colors: name -> a couple of representative RGB anchors used only by the
# OFFLINE fallback backend for nearest-color-name lookup. The CLIP backend
# does not need the RGB values, only the names (used to build text prompts).
# ---------------------------------------------------------------------------
COLORS = {
    "black":  (20, 20, 20),
    "white":  (235, 235, 235),
    "grey":   (128, 128, 128),
    "red":    (200, 30, 30),
    "maroon": (128, 0, 0),
    "pink":   (230, 150, 180),
    "orange": (230, 120, 30),
    "yellow": (230, 210, 40),
    "green":  (40, 130, 60),
    "olive":  (110, 110, 40),
    "blue":   (40, 80, 200),
    "navy":   (20, 30, 90),
    "teal":   (30, 140, 140),
    "purple": (120, 50, 150),
    "brown":  (120, 80, 45),
    "beige":  (215, 195, 155),
}

# Garment types, grouped by the body region they typically occupy.
# This grouping is what lets us bind "color" to "garment" per-region instead
# of treating a caption as an unordered bag of words.
GARMENTS_BY_REGION = {
    "upper_body": [
        "shirt", "button-down shirt", "blouse", "t-shirt", "hoodie",
        "sweater", "sweatshirt", "tank top", "polo shirt", "tie",
    ],
    "lower_body": [
        "pants", "trousers", "jeans", "shorts", "skirt", "leggings",
    ],
    "outerwear": [
        "blazer", "suit jacket", "coat", "raincoat", "trench coat",
        "denim jacket", "leather jacket", "puffer jacket", "cardigan",
    ],
    "full_body": [
        "dress", "jumpsuit", "suit",
    ],
}

ALL_GARMENTS = sorted({g for gs in GARMENTS_BY_REGION.values() for g in gs})

LOCATIONS = [
    "modern office interior",
    "city street",
    "urban sidewalk",
    "public park",
    "home / indoor living room",
    "studio background",
]

# Coarser labels used for matching against free-text queries
LOCATION_KEYWORDS = {
    "office":  ["office", "workplace", "desk", "corporate", "boardroom"],
    "street":  ["street", "city", "urban", "sidewalk", "downtown", "walk"],
    "park":    ["park", "bench", "outdoor", "garden", "trees"],
    "home":    ["home", "house", "living room", "indoor", "apartment"],
    "studio":  ["studio", "plain background", "catalog", "product shot"],
}

STYLES = [
    "formal business attire",
    "casual everyday wear",
    "athletic / sportswear",
    "outdoor / weatherproof wear",
    "evening / party wear",
]

STYLE_KEYWORDS = {
    "formal":  ["formal", "business", "professional", "office wear", "suit"],
    "casual":  ["casual", "weekend", "relaxed", "everyday", "streetwear"],
    "athletic":["athletic", "sport", "gym", "running", "activewear"],
    "weather": ["raincoat", "rain", "weatherproof", "winter", "cold", "snow"],
    "evening": ["evening", "party", "cocktail", "night out"],
}

# Region prompt templates used for zero-shot CLIP classification.
# "a photo of a {color} {garment}" is intentionally simple -- fashion-CLIP
# checkpoints (e.g. patrickjohncyh/fashion-clip, Marqo/marqo-fashionCLIP)
# are fine-tuned on exactly this style of short product caption.
PROMPT_TEMPLATE = "a photo of a person wearing a {color} {garment}"
LOCATION_PROMPT_TEMPLATE = "a photo taken in a {location}"
STYLE_PROMPT_TEMPLATE = "a photo of {style}"

# Embedding backend selection: "clip" (production) or "offline" (classical
# CV fallback that needs no GPU / network -- see indexer/embedding_backend.py)
DEFAULT_BACKEND = "offline"

# Fashion-domain CLIP checkpoint to use in production mode.
# patrickjohncyh/fashion-clip is a CLIP ViT-B/32 fine-tuned on ~800K
# fashion product image-caption pairs -- it is the recommended baseline
# encoder referenced in the write-up.
CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"

TOP_K_CANDIDATES = 100   # ANN candidates pulled before re-ranking
EMBED_DIM_OFFLINE = 64   # dimensionality of the offline fallback embedding
