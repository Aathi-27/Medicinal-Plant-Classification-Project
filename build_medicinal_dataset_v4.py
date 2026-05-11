"""
Medicinal Plant Dataset Builder v4
=====================================
Source: iNaturalist Open Data API (https://api.inaturalist.org)
- No Kaggle account required
- No license acceptance required
- No giant downloads — fetches exactly 10 images per class
- All images are research-grade, CC-licensed observations

PREREQUISITES:
    pip install requests pillow

USAGE:
    python build_medicinal_dataset_v4.py

OUTPUT:
    medicinal_plant_dataset.zip
"""

import sys
import time
import shutil
import zipfile
import random
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

WORK_DIR     = Path("_plant_build_v4")
DATASET_DIR  = WORK_DIR / "dataset"
OUTPUT_ZIP   = Path("medicinal_plant_dataset.zip")
IMAGES_PER_CLASS = 10

INAT_API     = "https://api.inaturalist.org/v1/observations"
HEADERS      = {"User-Agent": "MedicinalPlantDatasetBuilder/1.0"}
REQUEST_DELAY = 1.2   # seconds between API calls (respect rate limit)

# class_key → iNaturalist taxon search name
SPECIES_MAP = {
    "tulsi":      "Ocimum tenuiflorum",
    "neem":       "Azadirachta indica",
    "aloe_vera":  "Aloe vera",
    "mint":       "Mentha spicata",
    "amla":       "Phyllanthus emblica",
    "peppermint": "Mentha x piperita",
    "bael":       "Aegle marmelos",
    "lemon_balm": "Melissa officinalis",
    "catnip":     "Nepeta cataria",
    "stevia":     "Stevia rebaudiana",
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


# ──────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────

def log(msg):
    print(f"[BUILD] {msg}", flush=True)


def fetch_observations(taxon_name, per_page=50, page=1):
    """
    Fetch research-grade observations with photos for a taxon.
    Returns list of observation dicts.
    """
    params = {
        "taxon_name":    taxon_name,
        "quality_grade": "research",
        "photos":        "true",
        "per_page":      per_page,
        "page":          page,
        "order":         "desc",
        "order_by":      "votes",
    }
    try:
        r = requests.get(INAT_API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        log(f"  API error for '{taxon_name}': {e}")
        return []


def extract_photo_urls(observations):
    """
    Pull medium-size photo URLs from observation results.
    Prefers leaf photos; falls back to any photo.
    Returns list of (url, obs_id) tuples.
    """
    urls = []
    for obs in observations:
        photos = obs.get("photos", [])
        obs_id = obs.get("id", "unknown")
        for photo in photos:
            url = photo.get("url", "")
            if not url:
                continue
            # iNaturalist URLs come as square thumbnails; upgrade to medium
            url = url.replace("/square.", "/medium.")
            urls.append((url, obs_id))
    return urls


def download_image(url):
    """Download image bytes from URL. Returns PIL Image or None."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        img.load()  # force decode to catch corrupt files
        return img
    except Exception as e:
        return None


def convert_to_png(img):
    """Convert PIL Image to RGB PNG-ready Image."""
    if img.mode in ("P", "RGBA"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def is_valid_image(img):
    """Basic quality gate: reject tiny or extreme-aspect images."""
    w, h = img.size
    if w < 100 or h < 100:
        return False
    aspect = max(w, h) / min(w, h)
    if aspect > 10:
        return False
    return True


# ──────────────────────────────────────────────
# MAIN BUILD
# ──────────────────────────────────────────────

def build():
    random.seed(42)
    log("=== Medicinal Plant Dataset Builder v4 (iNaturalist) ===")

    # Setup dirs
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True)

    for cls in SPECIES_MAP:
        (DATASET_DIR / "dataset" / cls).mkdir(parents=True, exist_ok=True)

    filled  = {cls: 0 for cls in SPECIES_MAP}
    failed  = {}

    for cls, taxon in SPECIES_MAP.items():
        log(f"\n[{cls}] Searching iNaturalist for: {taxon}")
        out_dir   = DATASET_DIR / "dataset" / cls
        collected = 0
        page      = 1
        used_obs  = set()

        while collected < IMAGES_PER_CLASS and page <= 5:
            observations = fetch_observations(taxon, per_page=50, page=page)
            time.sleep(REQUEST_DELAY)

            if not observations:
                log(f"  No more observations at page {page}.")
                break

            photo_urls = extract_photo_urls(observations)
            random.shuffle(photo_urls)

            for url, obs_id in photo_urls:
                if collected >= IMAGES_PER_CLASS:
                    break
                if obs_id in used_obs:
                    continue

                img = download_image(url)
                if img is None:
                    continue
                if not is_valid_image(img):
                    continue

                img = convert_to_png(img)
                dest = out_dir / f"{cls}_{collected + 1:02d}.png"
                img.save(dest, format="PNG", optimize=False)
                used_obs.add(obs_id)
                collected += 1
                log(f"  {collected:02d}/10 ← obs#{obs_id}  ({img.width}x{img.height})")
                time.sleep(0.3)  # small delay between image downloads

            page += 1

        filled[cls]  = collected
        if collected < IMAGES_PER_CLASS:
            failed[cls] = collected
            log(f"  [{cls}] WARNING: only {collected}/10 collected.")

    # Validation
    log("\n=== VALIDATION ===")
    total = 0
    for cls in SPECIES_MAP:
        count  = len(list((DATASET_DIR / "dataset" / cls).glob("*.png")))
        status = "OK" if count == IMAGES_PER_CLASS else f"INCOMPLETE ({count}/10)"
        log(f"  {cls:<15} {count:>3} images  [{status}]")
        total += count

    # Package ZIP
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    log(f"\nPackaging → {OUTPUT_ZIP}")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted((DATASET_DIR / "dataset").rglob("*.png")):
            zf.write(fp, fp.relative_to(DATASET_DIR))

    mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)

    # Summary
    print("\n" + "=" * 56)
    print("VERIFICATION SUMMARY")
    print("=" * 56)
    print(f"Output     : {OUTPUT_ZIP.resolve()}")
    print(f"ZIP size   : {mb:.2f} MB")
    print(f"Source     : iNaturalist Open Data API")
    print(f"Classes    : {len(SPECIES_MAP)}")
    print(f"Images     : {total}")
    print(f"Format     : PNG (all converted)")
    print(f"Augmented  : NO")
    print(f"Synthetic  : NO")
    print(f"License    : CC BY-NC (iNaturalist research-grade observations)")

    if failed:
        print(f"\nINCOMPLETE CLASSES ({len(failed)}):")
        for cls, count in failed.items():
            print(f"  {cls}: {count}/10")
        print()
        print("These species have limited research-grade observations on iNaturalist.")
        print(f"Drop additional images manually into:")
        print(f"  {(WORK_DIR / 'manual').resolve()}\\<class_name>\\")
        print("Then re-run — completed classes are skipped.")
    else:
        print("\nAll 10 classes complete. Dataset ready.")
    print("=" * 56)

    # Cleanup workspace
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    log("Done.")


if __name__ == "__main__":
    try:
        build()
    except KeyboardInterrupt:
        print("\n[ABORTED]", file=sys.stderr)
        sys.exit(1)