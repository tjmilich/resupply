"""The full analysis, shared by the command line and the web app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .analysis import MERGE_M, Gap, Stop, find_gaps, locate, merge_nearby, pois_from_elements
from .categories import CATEGORIES, FOOD, WATER, Category
from .distances import DEFAULT_RADIUS_M
from .overpass import OverpassClient, fetch_elements
from .route import Route, chunk_coords, chunk_ranges, simplified_coords

if TYPE_CHECKING:
    from .extract import LocalExtract


@dataclass
class Options:
    categories: list[Category] = field(default_factory=lambda: list(CATEGORIES.values()))
    radii: dict[str, float] | None = None  # {category key: metres}; None = DEFAULT_RADIUS_M for all
    gap_limits: dict[str, float] = field(default_factory=lambda: {WATER: 50_000.0, FOOD: 50_000.0})  # 0 = off
    fuel_as_food: bool = True
    merge_distance_m: float = MERGE_M
    simplify_m: float = 50.0
    chunk_points: int = 150
    chunk_km: float = 50.0
    refresh: bool = False

    def radius_map(self) -> dict[str, float]:
        base = self.radii or {}
        return {c.key: base.get(c.key, DEFAULT_RADIUS_M) for c in self.categories}


@dataclass
class Result:
    route: Route
    stops: list[Stop]
    gaps: list[Gap]
    missing: list[tuple[float, float]]  # along-route stretches with no usable Overpass data
    n_chunks: int
    n_features: int
    n_local_chunks: int = 0  # chunks answered from the pre-built extract instead of Overpass
    extract_date: str = ""

    @property
    def n_places(self) -> int:
        return len({(s.poi.osm_type, s.poi.osm_id) for s in self.stops})


def plan_chunks(route: Route, opts: Options) -> list:
    coords = simplified_coords(route, opts.simplify_m)
    return chunk_coords(coords, opts.chunk_points, opts.chunk_km * 1000)


def analyze(
    route: Route,
    opts: Options,
    client: OverpassClient,
    on_progress: Callable[[int, int], None] | None = None,
    extract: "LocalExtract | None" = None,
) -> Result:
    """Find resupply points along ``route`` and build stops, gaps and no-data stretches.

    Chunks whose whole search corridor lies inside ``extract`` are answered locally;
    the rest go to Overpass. ``on_progress(done, total)`` is called after each chunk.
    Raises OverpassError only if every live chunk fails.
    """
    radii = opts.radius_map()
    chunks = plan_chunks(route, opts)
    # widen the corridor by the simplification tolerance; filter on the true route afterwards
    query_radii = {k: r + opts.simplify_m for k, r in radii.items()}
    widest = max(query_radii.values())
    local = [i for i, c in enumerate(chunks) if extract is not None and extract.covers(c, widest)]
    remote = [i for i in range(len(chunks)) if i not in set(local)]

    seen: dict[tuple[str, int], dict] = {}
    for n, i in enumerate(local, 1):
        for el in extract.elements_near(chunks[i], widest):
            seen.setdefault((el["type"], el["id"]), el)
        if on_progress:
            on_progress(n, len(chunks))

    missing_idx: list[int] = []
    if remote:
        live, miss = fetch_elements(
            client,
            [chunks[i] for i in remote],
            query_radii,
            opts.categories,
            refresh=opts.refresh,
            on_progress=(lambda d, n: on_progress(len(local) + d, len(chunks))) if on_progress else None,
        )
        for el in live:
            seen.setdefault((el["type"], el["id"]), el)
        missing_idx = [remote[j] for j in miss]
    elements = list(seen.values())
    ranges = chunk_ranges(chunks, route.length_m)
    missing = merge_ranges([ranges[i] for i in missing_idx])

    pois = pois_from_elements(elements, opts.categories, fuel_as_food=opts.fuel_as_food)
    stops = merge_nearby(locate(route, pois, radii), opts.merge_distance_m)
    gaps = sorted(
        (
            g
            for res in (WATER, FOOD)
            if opts.gap_limits.get(res, 0) > 0
            for g in find_gaps(route.length_m, stops, res, opts.gap_limits[res])
        ),
        key=lambda g: (g.start_m, g.resource),
    )
    for g in gaps:
        g.uncertain = any(a < g.end_m and b > g.start_m for a, b in missing)
    return Result(
        route, stops, gaps, missing, len(chunks), len(elements),
        n_local_chunks=len(local), extract_date=extract.data_date if (extract and local) else "",
    )


def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged
