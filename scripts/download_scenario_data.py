#!/usr/bin/env python3
"""Download three-scenario data: portraits, architecture, typography."""

import os
import json
import urllib.request
import urllib.error
import time
from pathlib import Path

BASE_DIR = Path("/home/hiaskc/Talant/graduation/data")

# Categories and their search queries for Pexels API
CATEGORIES = {
    "portraits": {
        "queries": ["portrait face person", "close-up face headshot"],
        "count": 8,
    },
    "architecture": {
        "queries": ["modern architecture building glass exterior"],
        "count": 5,
    },
    "typography": {
        "queries": ["typography lettering text sign", "neon sign light text"],
        "count": 5,
    },
}

PEXELS_API = "https://api.pexels.com/v1/search?query={query}&per_page={per_page}"


def search_pexels(query: str, per_page: int) -> list[dict]:
    """Search Pexels API and return photo list."""
    url = PEXELS_API.format(query=urllib.request.quote(query), per_page=per_page)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("photos", [])
    except urllib.error.URLError as e:
        print(f"  API error for '{query}': {e}")
        return []


def download_image(url: str, dest: Path) -> bool:
    """Download an image from URL to dest. Returns True on success."""
    if dest.exists():
        print(f"  Skip existing: {dest.name}")
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) < 1000:
                print(f"  Too small ({len(data)} bytes): {url}")
                return False
            dest.write_bytes(data)
            print(f"  Downloaded: {dest.name} ({len(data)} bytes)")
            return True
    except urllib.error.URLError as e:
        print(f"  Download error: {e}")
        return False


def main():
    for category, cfg in CATEGORIES.items():
        dest_dir = BASE_DIR / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Category: {category} (target: {cfg['count']} images)")
        print(f"{'='*60}")

        all_photos = []
        seen_ids = set()

        for query in cfg["queries"]:
            per_page = min(cfg["count"] + 2, 15)
            photos = search_pexels(query, per_page)
            for p in photos:
                if p["id"] not in seen_ids:
                    seen_ids.add(p["id"])
                    all_photos.append(p)
            print(f"  Query '{query}': found {len(photos)} photos")
            time.sleep(0.5)

        print(f"  Total unique photos: {len(all_photos)}")

        downloaded = 0
        for i, photo in enumerate(all_photos):
            if downloaded >= cfg["count"]:
                break

            # Use medium size for good quality at reasonable file size
            img_url = photo["src"]["large"]
            ext = ".jpeg" if "jpeg" in img_url else ".jpg"
            dest = dest_dir / f"pexels_{photo['id']}{ext}"

            print(f"  [{downloaded+1}/{cfg['count']}] {photo['alt'][:80]}...")
            if download_image(img_url, dest):
                downloaded += 1
                # Save metadata
                meta = {
                    "id": photo["id"],
                    "photographer": photo["photographer"],
                    "url": photo["url"],
                    "alt": photo["alt"],
                    "avg_color": photo.get("avg_color", ""),
                    "width": photo["width"],
                    "height": photo["height"],
                }
                meta_path = dest.with_suffix(".json")
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            time.sleep(0.3)

        print(f"  Downloaded: {downloaded}/{cfg['count']}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for category in CATEGORIES:
        d = BASE_DIR / category
        images = list(d.glob("*.jp*"))
        print(f"  {category}: {len(images)} images")
        for img in sorted(images):
            size_kb = img.stat().st_size / 1024
            print(f"    - {img.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
