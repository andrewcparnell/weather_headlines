"""Automatically compute content-hash `?v=` cache-busting strings across HTML and ES modules."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .config import SITE_DIR


def compute_frontend_asset_version(site_dir: Path = SITE_DIR) -> str:
    """Compute a deterministic 8-hex digest over `style.css` and `js/*.js` (ignoring existing `?v=` tokens)."""
    hasher = hashlib.sha256()
    paths = [
        site_dir / "style.css",
        *sorted((site_dir / "js").glob("*.js")),
    ]
    for p in paths:
        if not p.exists():
            continue
        normalised = re.sub(r"\?v=[A-Za-z0-9_.-]+", "?v=HASH", p.read_text())
        hasher.update(p.name.encode("utf-8"))
        hasher.update(normalised.encode("utf-8"))
    return hasher.hexdigest()[:8]


def stamp_frontend_assets(site_dir: Path = SITE_DIR) -> str:
    """Update `?v=<hash>` in `index.html` and `js/*.js` in a single automated pass."""
    ver = compute_frontend_asset_version(site_dir)
    targets = [
        site_dir / "index.html",
        *sorted((site_dir / "js").glob("*.js")),
    ]
    pat = re.compile(r"((?:style\.css|(?:js|\.)/[A-Za-z0-9_-]+\.js))\?v=[A-Za-z0-9_.-]+")
    for path in targets:
        if not path.exists():
            continue
        old_txt = path.read_text()
        new_txt = pat.sub(rf"\1?v={ver}", old_txt)
        if new_txt != old_txt:
            path.write_text(new_txt)
    return ver


if __name__ == "__main__":
    print(stamp_frontend_assets())
