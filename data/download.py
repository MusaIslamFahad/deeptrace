"""
data/download.py
Bootstrap the multi-generator dataset from public HuggingFace datasets.

Usage:
    python data/download.py --output data/raw --samples-per-class 2000

Sources:
    - stable_diffusion : datasets from HuggingFace (various SD checkpoints)
    - midjourney       : haor/Midjourney-v5-188K
    - dalle3           : phantom-nikon/dalle3-images (and similar)
    - flux             : artificialguybr/FluxDevImages or similar
    - real             : LAION subset from datasets, or COCO val images

NOTE:
    Some datasets require HuggingFace login: `huggingface-cli login`
    Real images: download MS COCO 2017 val from https://cocodataset.org/
    or use the Unsplash Lite dataset.
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from tqdm import tqdm


CATEGORY_KEYWORDS = {
    "animals":  ["cat", "dog", "bird", "animal", "wildlife"],
    "city":     ["city", "urban", "street", "building", "architecture"],
    "food":     ["food", "meal", "dish", "restaurant", "cuisine"],
    "nature":   ["nature", "landscape", "forest", "mountain", "outdoor"],
    "people":   ["person", "portrait", "face", "people", "human"],
}

HF_SOURCES = {
    "stable_diffusion": "poloclub/diffusiondb",
    "midjourney":       "haor/Midjourney-v5-188K",
    "real":             "nlphuji/flickr30k",
}

CATEGORY_NAMES = list(CATEGORY_KEYWORDS.keys())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_and_save(img: Image.Image, out_path: Path, size: int = 512):
    """Resize to at least `size` on shortest side, save as JPEG."""
    w, h = img.size
    if min(w, h) > size:
        scale = size / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = img.convert("RGB")
    img.save(out_path, "JPEG", quality=92)


def assign_category(text: str) -> str:
    """Assign one of the 5 categories based on keyword matching."""
    text_lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return cat
    return "nature"  # default fallback


def download_hf_source(
    source_name: str,
    dataset_id: str,
    output_root: Path,
    samples_per_class: int,
    image_col: str = "image",
    text_col: Optional[str] = "text",
    split: str = "train",
):
    """Download images from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Install datasets: pip install datasets")
        return

    print(f"\n[download] {source_name} ← {dataset_id}")

    # Track how many images we have per category
    category_counts = {cat: 0 for cat in CATEGORY_NAMES}
    total_needed = samples_per_class * len(CATEGORY_NAMES)

    # Create output dirs
    for cat in CATEGORY_NAMES:
        ensure_dir(output_root / source_name / cat)

    try:
        dataset = load_dataset(dataset_id, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  Could not load {dataset_id}: {e}")
        return

    saved = 0
    pbar = tqdm(total=total_needed, desc=f"  {source_name}")

    for item in dataset:
        if saved >= total_needed:
            break

        # Check if all categories are full
        if all(v >= samples_per_class for v in category_counts.values()):
            break

        # Get image
        img = item.get(image_col)
        if img is None:
            continue
        if isinstance(img, dict):  # HF image dict with bytes
            try:
                from io import BytesIO
                img = Image.open(BytesIO(img.get("bytes", b"")))
            except Exception:
                continue
        if not isinstance(img, Image.Image):
            continue

        # Determine category from text/caption
        text = ""
        if text_col and text_col in item:
            text = str(item[text_col])

        category = assign_category(text)
        if category_counts[category] >= samples_per_class:
            continue

        # Save
        idx = category_counts[category]
        out_path = output_root / source_name / category / f"{source_name}_{category}_{idx:05d}.jpg"
        try:
            resize_and_save(img, out_path)
            category_counts[category] += 1
            saved += 1
            pbar.update(1)
        except Exception:
            pass

    pbar.close()
    print(f"  Saved: {category_counts}")


def download_real_coco(output_root: Path, samples_per_class: int):
    """
    Download COCO 2017 validation images (5000 images, free).
    Falls back to a small Flickr30k sample if COCO unavailable.
    """
    try:
        import urllib.request
        from io import BytesIO
        import json as _json

        print("\n[download] real ← COCO 2017 val")

        coco_ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        print("  Downloading COCO annotations...")

        # Use HF flickr30k as easier alternative
        download_hf_source(
            source_name="real",
            dataset_id="nlphuji/flickr30k",
            output_root=output_root,
            samples_per_class=samples_per_class,
            image_col="image",
            text_col="caption",
        )

    except Exception as e:
        print(f"  Real image download failed: {e}")
        print("  Please manually place real images in data/raw/real/<category>/")


def create_dummy_dataset(output_root: Path, samples_per_class: int = 20):
    """
    Create a tiny synthetic dataset for testing the pipeline without downloads.
    Generates solid-color JPEG images labelled by source and category.
    """
    import numpy as np

    SOURCES = ["stable_diffusion", "midjourney", "dalle3", "flux", "real"]
    COLORS = {
        "stable_diffusion": (180, 140, 255),
        "midjourney":       (100, 220, 170),
        "dalle3":           (255, 150, 100),
        "flux":             (100, 180, 255),
        "real":             (140, 200, 100),
    }

    print("\n[download] Creating dummy dataset for pipeline testing...")
    for source in SOURCES:
        for cat in CATEGORY_NAMES:
            out_dir = ensure_dir(output_root / source / cat)
            for i in range(samples_per_class):
                # Add slight noise so images aren't identical
                color = tuple(
                    min(255, max(0, c + np.random.randint(-20, 20)))
                    for c in COLORS[source]
                )
                img_array = np.full((224, 224, 3), color, dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(out_dir / f"dummy_{i:04d}.jpg", "JPEG")

    total = len(SOURCES) * len(CATEGORY_NAMES) * samples_per_class
    print(f"  Created {total} dummy images in {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Download DeepTrace dataset")
    parser.add_argument("--output", default="data/raw",
                        help="Output directory for raw images")
    parser.add_argument("--samples-per-class", type=int, default=2000,
                        help="Target images per (source, category) combination")
    parser.add_argument("--dummy", action="store_true",
                        help="Create tiny dummy dataset for pipeline testing (no downloads)")
    parser.add_argument("--sources", nargs="+",
                        default=["stable_diffusion", "midjourney", "real"],
                        help="Which sources to download")
    args = parser.parse_args()

    output_root = Path(args.output)

    if args.dummy:
        create_dummy_dataset(output_root, samples_per_class=20)
        print("\n✓ Dummy dataset ready. Run: python -c \"from data.dataset import build_manifest; build_manifest()\"")
        return

    print(f"Downloading dataset to {output_root}")
    print(f"Target: {args.samples_per_class} images per (source × category)")

    if "stable_diffusion" in args.sources:
        download_hf_source(
            "stable_diffusion", "poloclub/diffusiondb",
            output_root, args.samples_per_class,
            image_col="image", text_col="prompt",
        )

    if "midjourney" in args.sources:
        download_hf_source(
            "midjourney", "haor/Midjourney-v5-188K",
            output_root, args.samples_per_class,
            image_col="image", text_col="text",
        )

    if "dalle3" in args.sources:
        download_hf_source(
            "dalle3", "nousr/laion-aesthetics-dalle3",
            output_root, args.samples_per_class,
            image_col="image", text_col="caption",
        )

    if "flux" in args.sources:
        download_hf_source(
            "flux", "artificialguybr/FluxDevImages",
            output_root, args.samples_per_class,
            image_col="image", text_col=None,
        )

    if "real" in args.sources:
        download_real_coco(output_root, args.samples_per_class)

    print(f"\n✓ Download complete. Now run:")
    print(f"  python -c \"from data.dataset import build_manifest; build_manifest()\"")
    print(f"  dvc add data/raw data/manifest.csv")


if __name__ == "__main__":
    main()
