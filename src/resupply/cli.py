"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .analysis import find_gaps, locate, pois_from_elements
from .categories import CATEGORIES, FOOD, WATER, parse_category_list
from .output import KM, MI, gap_message, missing_message, write_csv, write_map
from .overpass import DEFAULT_ENDPOINTS, DiskCache, OverpassClient, OverpassError, default_cache_dir, fetch_elements
from .route import RouteError, chunk_coords, chunk_ranges, load_gpx, simplified_coords

log = logging.getLogger("resupply")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="resupply",
        description="Find water, food, fuel, campsites and bike shops along a GPX route (OpenStreetMap/Overpass).",
    )
    p.add_argument("gpx", type=Path, help="input GPX file (tracks, or routes if no tracks)")
    p.add_argument("-o", "--out-dir", type=Path, help="output directory (default: next to the GPX)")
    p.add_argument("--map", type=Path, help="map HTML path (default: <name>_resupply.html)")
    p.add_argument("--csv", type=Path, help="cue sheet CSV path (default: <name>_cues.csv)")

    g = p.add_argument_group("search")
    g.add_argument("-r", "--radius", type=float, default=1000.0,
                   help="max offset from route in metres (default: 1000)")
    g.add_argument("-c", "--categories", default="all",
                   help=f"comma-separated subset of: {', '.join(CATEGORIES)} (default: all)")
    g.add_argument("--no-fuel-food", action="store_true",
                   help="don't count gas stations as food/water unless tagged with a shop")

    g = p.add_argument_group("gaps")
    g.add_argument("--units", choices=["km", "mi"], default="km", help="distance units for output (default: km)")
    g.add_argument("--max-gap", type=float,
                   help="flag stretches longer than this with no water or no food (default: 50 km / 30 mi)")
    g.add_argument("--max-water-gap", type=float, help="override --max-gap for water")
    g.add_argument("--max-food-gap", type=float, help="override --max-gap for food")

    g = p.add_argument_group("query tuning")
    g.add_argument("--simplify", type=float, default=50.0,
                   help="route simplification tolerance in metres for queries (default: 50)")
    g.add_argument("--chunk-points", type=int, default=150, help="max vertices per Overpass query (default: 150)")
    g.add_argument("--chunk-km", type=float, default=50.0, help="max route length per Overpass query (default: 50)")
    g.add_argument("--endpoint", action="append", help="Overpass endpoint URL (repeatable; default: public servers)")

    g = p.add_argument_group("cache")
    g.add_argument("--cache-dir", type=Path, default=None, help=f"cache directory (default: {default_cache_dir()})")
    g.add_argument("--cache-ttl-days", type=float, default=30.0, help="cache lifetime; 0 = never expire (default: 30)")
    g.add_argument("--refresh", action="store_true", help="ignore cached responses and re-query (updates cache)")
    g.add_argument("--no-cache", action="store_true", help="don't read or write the cache")

    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("-q", "--quiet", action="store_true", help="only print warnings and errors")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    if args.radius <= 0:
        parser.error("--radius must be positive")
    if args.chunk_points < 2:
        parser.error("--chunk-points must be at least 2")
    try:
        categories = parse_category_list(args.categories)
    except ValueError as e:
        parser.error(str(e))

    units = MI if args.units == "mi" else KM
    max_gap = args.max_gap if args.max_gap is not None else (30.0 if units is MI else 50.0)
    gap_limits = {
        WATER: (args.max_water_gap if args.max_water_gap is not None else max_gap) * units.long_m,
        FOOD: (args.max_food_gap if args.max_food_gap is not None else max_gap) * units.long_m,
    }

    out_dir = args.out_dir or args.gpx.parent
    stem = args.gpx.stem
    map_path = args.map or out_dir / f"{stem}_resupply.html"
    csv_path = args.csv or out_dir / f"{stem}_cues.csv"

    try:
        route = load_gpx(args.gpx)
        log.info("Route '%s': %s %s, %d points", route.name, units.dist(route.length_m), units.long, len(route))

        coords = simplified_coords(route, args.simplify)
        chunks = chunk_coords(coords, args.chunk_points, args.chunk_km * 1000)
        log.info("Simplified to %d points -> %d Overpass queries", len(coords), len(chunks))

        cache = None
        if not args.no_cache:
            ttl = args.cache_ttl_days * 86400 if args.cache_ttl_days > 0 else None
            cache = DiskCache(args.cache_dir or default_cache_dir(), ttl_s=ttl)
        client = OverpassClient(endpoints=args.endpoint or DEFAULT_ENDPOINTS, cache=cache)
        # widen the query corridor by the simplification tolerance; filter on the true route afterwards
        elements, missing_idx = fetch_elements(
            client, chunks, args.radius + args.simplify, categories, refresh=args.refresh
        )
    except RouteError as e:
        log.error("error: %s", e)
        return 1
    except OverpassError as e:
        log.error("error: %s", e)
        return 1

    ranges = chunk_ranges(chunks, route.length_m)
    missing = _merge_ranges([ranges[i] for i in missing_idx])

    pois = pois_from_elements(elements, categories, fuel_as_food=not args.no_fuel_food)
    stops = locate(route, pois, args.radius)
    gaps = sorted(
        (g for res in (WATER, FOOD) if gap_limits[res] > 0 for g in find_gaps(route.length_m, stops, res, gap_limits[res])),
        key=lambda g: (g.start_m, g.resource),
    )
    for g in gaps:
        g.uncertain = any(a < g.end_m and b > g.start_m for a, b in missing)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, stops, gaps, units, missing)
    write_map(map_path, route, stops, gaps, units, missing)

    counts = Counter(s.poi.category.label for s in stops)
    unique = len({(s.poi.osm_type, s.poi.osm_id) for s in stops})
    log.info("Found %d resupply points (%d cue entries): %s", unique, len(stops),
             ", ".join(f"{k} {v}" for k, v in counts.most_common()) or "none")
    if gaps:
        log.warning("%d gap(s) flagged:", len(gaps))
        for g in gaps:
            log.warning("  %s %s: %s", units.long, units.dist(g.start_m), gap_message(g, units))
    else:
        log.info("No gaps over the configured limits.")
    log.info("Map:       %s", map_path)
    log.info("Cue sheet: %s", csv_path)
    if missing:
        log.warning("WARNING: no usable map data for %d stretch(es); results there are incomplete:", len(missing))
        for a, b in missing:
            log.warning("  %s", missing_message(a, b, units))
        return EXIT_INCOMPLETE
    return 0


EXIT_INCOMPLETE = 3


def _merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


if __name__ == "__main__":
    sys.exit(main())
