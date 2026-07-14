"""
eval/run_eval_queries.py
--------------------------
Runs the 5 evaluation queries from the assignment brief against a built
index, saves a JSON results file and a PNG thumbnail grid per query
(for inclusion in the submission write-up / manual visual inspection).

Usage:
    python -m eval.run_eval_queries --index outputs/index --backend offline --out eval/results
"""
from __future__ import annotations
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retriever.search import search  # noqa: E402

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def make_grid(paths, scores, out_path, thumb=(220, 220)):
    n = len(paths)
    grid = Image.new("RGB", (thumb[0] * n, thumb[1] + 30), "white")
    draw = ImageDraw.Draw(grid)
    for i, (p, s) in enumerate(zip(paths, scores)):
        img = Image.open(p).convert("RGB").resize(thumb)
        grid.paste(img, (i * thumb[0], 30))
        draw.text((i * thumb[0] + 5, 5), f"#{i+1} score={s:.2f}", fill="black")
    grid.save(out_path)


def main(index_prefix: str, backend: str, out_dir: str, k: int = 5):
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}
    for qi, query in enumerate(EVAL_QUERIES, 1):
        hits = search(index_prefix, query, backend, k=k)
        all_results[query] = hits
        grid_path = os.path.join(out_dir, f"query_{qi}.png")
        make_grid([h["path"] for h in hits], [h["final_score"] for h in hits], grid_path)
        print(f"[{qi}] {query}")
        for h in hits:
            print(f"     {h['image_id']}  final={h['final_score']}  embed={h['embed_score']}  attr={h['attr_score']}  {h['explanations']}")
        print(f"     -> grid saved to {grid_path}")

    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--backend", default="offline", choices=["clip", "offline"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(args.index, args.backend, args.out, args.k)
