"""Pre-built OpenStreetMap extracts: build them from Geofabrik files, and read them.

Building (needs the optional ``osmium`` package, ``pip install 'resupply[extract]'``):
for each region, download the Geofabrik ``.osm.pbf``, keep only features that match a
resupply category, write them into 1-degree tiles, and delete the download before the
next region. Peak disk use is therefore one regional file.

Reading (``LocalExtract``) needs nothing extra. It answers "is this stretch of route
inside the extract?" and "which features are near it?" so the pipeline can skip the
live Overpass API wherever the extract has coverage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import logging
import math
import shutil
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping, shape
from shapely.ops import unary_union

from . import __version__
from .categories import CATEGORIES, classify
from .route import LatLon

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
TILE_DEG = 1
GEOFABRIK = "https://download.geofabrik.de"
# Tags the rest of the tool reads (classification, provides, names, notes); everything else is dropped.
KEEP_TAGS = {
    "amenity", "shop", "tourism", "man_made", "natural", "drinking_water", "access",
    "name", "brand", "operator", "opening_hours", "fee", "seasonal", "phone",
}
_FILTER_KEYS = ("amenity", "shop", "tourism", "man_made", "natural")

REGION_SETS: dict[str, list[str]] = {
    "northeast": [
        f"north-america/us/{s}"
        for s in (
            "maine", "vermont", "new-hampshire", "massachusetts", "connecticut", "rhode-island",
            "new-york", "new-jersey", "pennsylvania", "delaware", "maryland", "district-of-columbia",
            "virginia",
        )
    ],
}

DEFAULT_EXTRACT_DIR = Path(__file__).resolve().parent / "data" / "extract"
M_PER_DEG = 111_320.0


# ------------------------------------------------------------------------------------------
# Reading


def _deg_for(metres: float, lat: float) -> float:
    """Generous metres -> degrees conversion (longitude degrees shrink with latitude)."""
    return metres / (M_PER_DEG * max(math.cos(math.radians(min(abs(lat), 85.0))), 0.05))


def tile_name(lat: float, lon: float) -> str:
    return f"{math.floor(lat / TILE_DEG) * TILE_DEG}_{math.floor(lon / TILE_DEG) * TILE_DEG}.jsonl.gz"


class LocalExtract:
    """A built extract on disk: ``manifest.json`` plus ``tiles/*.jsonl.gz``."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        with (self.dir / "manifest.json").open(encoding="utf-8") as f:
            self.manifest = json.load(f)
        if self.manifest.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported extract format in {self.dir}")
        self.coverage = shape(self.manifest["coverage"])
        shapely.prepare(self.coverage)
        self.tiles = set(self.manifest["tiles"])

    @property
    def data_date(self) -> str:
        return self.manifest.get("osm_data_date") or self.manifest.get("built", "")[:10]

    @property
    def regions(self) -> list[str]:
        return self.manifest.get("regions", [])

    def _corridor(self, coords: list[LatLon], radius_m: float):
        lat_mid = float(np.mean([c[0] for c in coords]))
        geom = Point(coords[0][1], coords[0][0]) if len(coords) == 1 else LineString([(lon, lat) for lat, lon in coords])
        return geom, _deg_for(radius_m, lat_mid) * 1.2

    def covers(self, coords: list[LatLon], radius_m: float) -> bool:
        """True if the whole search corridor around this polyline lies inside the extract."""
        geom, d = self._corridor(coords, radius_m)
        return bool(self.coverage.contains(geom.buffer(d)))

    @lru_cache(maxsize=256)
    def _tile(self, name: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
        if name not in self.tiles:
            return [], np.empty(0), np.empty(0)
        with gzip.open(self.dir / "tiles" / name, "rt", encoding="utf-8") as f:
            els = [json.loads(line) for line in f]
        return els, np.array([e["lat"] for e in els]), np.array([e["lon"] for e in els])

    def elements_near(self, coords: list[LatLon], radius_m: float) -> list[dict]:
        """Features within (roughly, a superset of) ``radius_m`` of the polyline."""
        geom, d = self._corridor(coords, radius_m)
        minx, miny, maxx, maxy = geom.buffer(d).bounds
        out: list[dict] = []
        for lat in range(math.floor(miny / TILE_DEG) * TILE_DEG, math.floor(maxy / TILE_DEG) * TILE_DEG + 1, TILE_DEG):
            for lon in range(math.floor(minx / TILE_DEG) * TILE_DEG, math.floor(maxx / TILE_DEG) * TILE_DEG + 1, TILE_DEG):
                els, lats, lons = self._tile(f"{lat}_{lon}.jsonl.gz")
                if not els:
                    continue
                near = shapely.dwithin(shapely.points(lons, lats), geom, d)
                out.extend(e for e, k in zip(els, near) if k)
        return out


def find_extract(explicit: str | Path | None = None) -> LocalExtract | None:
    """The extract to use: an explicit path, $RESUPPLY_EXTRACT, or the one bundled with the package."""
    import os

    if explicit is not None:  # an explicit path is used alone; a typo must not fall back silently
        return LocalExtract(explicit) if (Path(explicit) / "manifest.json").exists() else None
    for candidate in (os.environ.get("RESUPPLY_EXTRACT"), DEFAULT_EXTRACT_DIR):
        if candidate and (Path(candidate) / "manifest.json").exists():
            return LocalExtract(candidate)
    return None


# ------------------------------------------------------------------------------------------
# Building


def parse_poly(text: str):
    """Parse an Osmosis .poly file (as published by Geofabrik) into a shapely geometry."""
    lines = [ln.strip() for ln in text.splitlines()]
    polys: list[Polygon] = []
    holes: list[Polygon] = []
    i = 1  # first line is the file's name
    while i < len(lines):
        name = lines[i]
        i += 1
        if name in ("", "END"):
            continue
        ring = []
        while i < len(lines) and lines[i] != "END":
            parts = lines[i].split()
            if len(parts) >= 2:
                ring.append((float(parts[0]), float(parts[1])))
            i += 1
        i += 1  # skip END
        if len(ring) >= 3:
            (holes if name.startswith("!") else polys).append(Polygon(ring))
    geom = unary_union(polys)
    if holes:
        geom = geom.difference(unary_union(holes))
    return geom


def extract_pbf(path: str | Path) -> list[dict]:
    """Features in a .osm.pbf that match any resupply category, as Overpass-style dicts.

    Nodes keep their position; ways use their bounding-box centre (as Overpass's
    ``out center`` does); multipolygon relations use their area's centroid.
    """
    import osmium  # optional dependency, only needed to build

    cats = list(CATEGORIES.values())
    out: list[dict] = []
    fp = (
        osmium.FileProcessor(str(path))
        .with_locations()
        .with_areas()
        .with_filter(osmium.filter.KeyFilter(*_FILTER_KEYS))
    )
    for obj in fp:
        tags = {t.k: t.v for t in obj.tags}
        if classify(tags, cats) is None:
            continue
        kept = {k: v for k, v in tags.items() if k in KEEP_TAGS}
        try:
            if obj.is_node():
                if not obj.location.valid():
                    continue
                out.append(_el("node", obj.id, obj.location.lat, obj.location.lon, kept))
            elif obj.is_way():
                locs = np.array([(n.lat, n.lon) for n in obj.nodes if n.location.valid()])
                if len(locs):
                    # bounding-box centre, exactly like Overpass's `out center`
                    la, lo = (locs.min(axis=0) + locs.max(axis=0)) / 2
                    out.append(_el("way", obj.id, la, lo, kept))
            elif obj.is_area() and not obj.from_way():
                rings = [Polygon([(n.lon, n.lat) for n in r]) for r in obj.outer_rings()]
                rings = [r for r in rings if r.is_valid and not r.is_empty]
                if rings:
                    c = MultiPolygon(rings).centroid
                    out.append(_el("relation", obj.orig_id(), c.y, c.x, kept))
        except osmium.InvalidLocationError:
            continue
    return out


def _el(osm_type: str, osm_id: int, lat: float, lon: float, tags: dict) -> dict:
    return {"type": osm_type, "id": int(osm_id), "lat": round(float(lat), 7), "lon": round(float(lon), 7), "tags": tags}


def _download(url: str, dest: Path, session: requests.Session) -> int:
    with session.get(url, stream=True, timeout=(15, 300), headers={"User-Agent": f"resupply-extract/{__version__}"}) as r:
        r.raise_for_status()
        size = 0
        with dest.open("wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
                size += len(block)
    return size


def write_extract(out_dir: Path, elements: list[dict], coverage, regions: list[str], data_date: str) -> dict:
    """Write tiles + manifest atomically (into a temp dir, then swap)."""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "tiles").mkdir(parents=True)
    by_tile: dict[str, list[dict]] = {}
    for e in elements:
        by_tile.setdefault(tile_name(e["lat"], e["lon"]), []).append(e)
    for name, els in sorted(by_tile.items()):
        els.sort(key=lambda e: (e["type"], e["id"]))  # deterministic output -> small git diffs
        with gzip.GzipFile(tmp / "tiles" / name, "wb", mtime=0) as gz:
            gz.write("".join(json.dumps(e, separators=(",", ":"), ensure_ascii=False) + "\n" for e in els).encode())
    manifest = {
        "format": FORMAT_VERSION,
        "tile_deg": TILE_DEG,
        "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "osm_data_date": data_date,
        "regions": regions,
        "feature_count": len(elements),
        "tiles": {k: len(v) for k, v in sorted(by_tile.items())},
        "coverage": mapping(coverage.simplify(0.001, preserve_topology=True)),
        "source": "OpenStreetMap contributors via Geofabrik (ODbL)",
    }
    with (tmp / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return manifest


def build(regions: list[str], out_dir: Path, work_dir: Path | None = None, keep_downloads: bool = False) -> dict:
    session = requests.Session()
    work = Path(work_dir or tempfile.mkdtemp(prefix="resupply-extract-"))
    work.mkdir(parents=True, exist_ok=True)
    seen: dict[tuple[str, int], dict] = {}
    shapes = []
    dates = []
    t0 = time.time()
    for k, region in enumerate(regions, 1):
        slug = region.rsplit("/", 1)[-1]
        region_dir = work / slug  # each download in its own empty directory
        region_dir.mkdir(exist_ok=True)
        pbf = region_dir / f"{slug}.osm.pbf"
        log.info("[%d/%d] %s: downloading...", k, len(regions), region)
        size = _download(f"{GEOFABRIK}/{region}-latest.osm.pbf", pbf, session)
        poly = session.get(f"{GEOFABRIK}/{region}.poly", timeout=60, headers={"User-Agent": f"resupply-extract/{__version__}"})
        poly.raise_for_status()
        shapes.append(parse_poly(poly.text))
        t = time.time()
        els = extract_pbf(pbf)
        for e in els:
            seen.setdefault((e["type"], e["id"]), e)  # neighbouring extracts overlap at borders
        try:
            import osmium

            ts = osmium.io.Reader(str(pbf), osmium.osm.osm_entity_bits.NOTHING).header().get("osmosis_replication_timestamp")
            if ts:
                dates.append(ts[:10])
        except Exception:  # header timestamp is a nice-to-have
            pass
        log.info("[%d/%d] %s: %.0f MB -> %d features (%.0fs)", k, len(regions), region, size / 1e6, len(els), time.time() - t)
        if not keep_downloads:
            shutil.rmtree(region_dir)
    manifest = write_extract(out_dir, list(seen.values()), unary_union(shapes), regions, min(dates) if dates else "")
    log.info("Wrote %d features in %d tiles to %s (%.0f min total)",
             manifest["feature_count"], len(manifest["tiles"]), out_dir, (time.time() - t0) / 60)
    if not keep_downloads and work_dir is None:
        shutil.rmtree(work, ignore_errors=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="resupply-extract", description="Build a pre-filtered OSM extract for resupply.")
    p.add_argument("regions", nargs="*", default=["northeast"],
                   help=f"Geofabrik region ids (e.g. north-america/us/vermont) or a set name: {', '.join(REGION_SETS)}")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_EXTRACT_DIR, help=f"output directory (default: {DEFAULT_EXTRACT_DIR})")
    p.add_argument("--work-dir", type=Path, help="where downloads go temporarily (default: a temp dir)")
    p.add_argument("--keep-downloads", action="store_true", help="don't delete the .osm.pbf files")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    regions = [r for item in args.regions for r in REGION_SETS.get(item, [item])]
    build(regions, args.out, args.work_dir, args.keep_downloads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
