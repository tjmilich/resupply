"""Turn OSM elements into located stops along a route, and find resupply gaps."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

import numpy as np
import shapely

from .categories import CATEGORIES, FOOD, WATER, Category, classify, provides
from .route import GEOD, Route

ON_ROUTE_M = 20.0  # offsets below this are reported as "on route" (no side)
MERGE_M = 60.0  # default distance for merging OSM features that are one real-world stop
_PRIORITY = {k: i for i, k in enumerate(CATEGORIES)}
# Categories that are often mapped as separate features of one business (a fuel station,
# its shop, the market inside). Anything else only merges with its own category, so a park
# fountain is never folded into the supermarket across the street.
_SHOP_LIKE = {"fuel", "convenience", "supermarket"}


@dataclass
class POI:
    osm_type: str
    osm_id: int
    lat: float
    lon: float
    tags: dict
    category: Category
    provides: frozenset

    @property
    def name(self) -> str:
        t = self.tags
        return t.get("name") or t.get("brand") or t.get("operator") or ""

    @property
    def osm_url(self) -> str:
        return f"https://www.openstreetmap.org/{self.osm_type}/{self.osm_id}"


@dataclass
class Stop:
    """One pass of the route by a real-world stop. A stop passed twice yields two Stops.

    ``poi`` is the primary OSM feature (its category sets the map icon and layer, its
    location is plotted). ``also`` holds other features merged into the same stop, e.g. a
    gas station's separately mapped convenience store.
    """

    poi: POI
    along_m: float
    offset_m: float
    side: str  # "L", "R" or "" when on the route
    next_water_m: float | None = None
    next_food_m: float | None = None
    also: list[POI] = field(default_factory=list)

    @property
    def pois(self) -> list[POI]:
        return [self.poi, *self.also]

    @property
    def provides(self) -> frozenset:
        return frozenset().union(*(p.provides for p in self.pois))

    @property
    def name(self) -> str:
        return next((p.name for p in self.pois if p.name), "")

    @property
    def categories(self) -> list[Category]:
        seen = {p.category.key: p.category for p in self.pois}
        return sorted(seen.values(), key=lambda c: _PRIORITY.get(c.key, 99))

    @property
    def label(self) -> str:
        return " + ".join(c.label for c in self.categories)

    def tag(self, key: str) -> str:
        """First non-empty value of a tag across merged features (primary first)."""
        return next((p.tags[key] for p in self.pois if p.tags.get(key)), "")


@dataclass
class Gap:
    resource: str
    start_m: float
    end_m: float
    at_start: bool = field(default=False)
    at_end: bool = field(default=False)
    uncertain: bool = field(default=False)  # overlaps a stretch with no Overpass data

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


def pois_from_elements(elements: list[dict], categories: list[Category], fuel_as_food: bool = True) -> list[POI]:
    pois = []
    for el in elements:
        tags = el.get("tags") or {}
        cat = classify(tags, categories)
        if cat is None:
            continue
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        pois.append(POI(el["type"], el["id"], float(lat), float(lon), tags, cat, provides(cat, tags, fuel_as_food)))
    return pois


def locate(route: Route, pois: list[POI], max_offset_m: float, pass_separation_m: float | None = None) -> list[Stop]:
    """Snap POIs to the full-resolution route.

    Candidate segments come from an STRtree ``dwithin`` query in the local
    projection (with a margin for projection distortion). Candidates are grouped
    into *passes*: runs of segments whose along-route distance is contiguous.
    Separate passes (out-and-back, loops, figure-eights) become separate stops.
    Final along-route distance and offset are geodesic.
    """
    if not pois:
        return []
    if pass_separation_m is None:
        pass_separation_m = max(2 * max_offset_m, 1000.0)

    xy = route.xy
    a_all, b_all = xy[:-1], xy[1:]
    tree = shapely.STRtree(shapely.linestrings(np.stack([a_all, b_all], axis=1)))
    px, py = route.to_xy([p.lat for p in pois], [p.lon for p in pois])
    search = max_offset_m * 1.25 + 50.0
    poi_idx, seg_idx = tree.query(shapely.points(px, py), predicate="dwithin", distance=search)

    order = np.lexsort((seg_idx, poi_idx))
    poi_idx, seg_idx = poi_idx[order], seg_idx[order]
    bounds = np.flatnonzero(np.diff(poi_idx)) + 1
    stops: list[Stop] = []

    for p_rows, s_rows in zip(np.split(poi_idx, bounds), np.split(seg_idx, bounds)):
        if len(p_rows) == 0:
            continue
        i = int(p_rows[0])
        poi = pois[i]
        p = np.array([px[i], py[i]])
        a, b = a_all[s_rows], b_all[s_rows]
        ab = b - a
        den = np.einsum("ij,ij->i", ab, ab)
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.where(den > 0, den, 1.0), 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist = np.hypot(*(p - proj).T)

        # split candidates into passes where along-route distance jumps
        jumps = route.cum_m[s_rows[1:]] - route.cum_m[s_rows[:-1] + 1] > pass_separation_m
        groups = np.split(np.arange(len(s_rows)), np.flatnonzero(jumps) + 1)
        for g in groups:
            j = g[np.argmin(dist[g])]
            seg = int(s_rows[j])
            slat, slon = route.to_latlon(proj[j, 0], proj[j, 1])
            _, _, offset = GEOD.inv(poi.lon, poi.lat, float(slon), float(slat))
            if offset > max_offset_m:
                continue
            along = float(route.cum_m[seg] + t[j] * route.seg_m[seg])
            cross = ab[j, 0] * (p[1] - a[j, 1]) - ab[j, 1] * (p[0] - a[j, 0])
            side = "" if offset < ON_ROUTE_M else ("L" if cross > 0 else "R")
            stops.append(Stop(poi, along, float(offset), side))

    stops.sort(key=lambda s: (s.along_m, s.offset_m))
    annotate_next(stops)
    return stops


def _norm_name(name: str) -> str:
    name = name.lower().replace("\u2019", "").replace("'", "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name).split())


def _same_business(a: str, b: str) -> bool:
    """Names match if equal or one extends the other ("sinclair" / "sinclair dinomart")."""
    return a == b or a.startswith(b + " ") or b.startswith(a + " ")


def _can_merge(cluster: list[Stop], s: Stop, distance_m: float) -> bool:
    first = cluster[0]
    if abs(s.along_m - first.along_m) > max(4 * distance_m, 500.0):
        return False  # different pass of the route
    cats = {c.poi.category.key for c in cluster}
    k = s.poi.category.key
    if not ((k in _SHOP_LIKE and cats <= _SHOP_LIKE) or cats == {k}):
        return False
    name = _norm_name(s.poi.name)
    if name and not all(_same_business(n, name) for n in (_norm_name(c.poi.name) for c in cluster) if n):
        return False
    _, _, d = GEOD.inv(
        [c.poi.lon for c in cluster], [c.poi.lat for c in cluster], [s.poi.lon] * len(cluster), [s.poi.lat] * len(cluster)
    )
    # near some member, and not too far from the first one, so a dense strip can't chain together
    return min(d) <= distance_m and d[0] <= 2 * distance_m


def merge_nearby(stops: list[Stop], distance_m: float = MERGE_M) -> list[Stop]:
    """Merge stops that are the same real-world place mapped as several OSM features.

    Typical cases: a fuel station and its convenience store, several nodes for one
    brand's pumps, unnamed pump nodes next to a named station, duplicate water taps.
    Separately named businesses never merge, nor do different passes of the route.
    """
    if distance_m <= 0 or len(stops) < 2:
        return list(stops)
    clusters: list[list[Stop]] = []
    for s in sorted(stops, key=lambda s: (s.along_m, s.offset_m)):
        target = None
        for c in reversed(clusters):
            if s.along_m - c[0].along_m > max(4 * distance_m, 500.0):
                break
            if _can_merge(c, s, distance_m):
                target = c
                break
        if target is None:
            clusters.append([s])
        else:
            target.append(s)

    merged = []
    for c in clusters:
        if len(c) == 1:
            merged.append(c[0])
            continue
        primary = min(c, key=lambda s: (_PRIORITY.get(s.poi.category.key, 99), not s.poi.name, s.offset_m))
        nearest = min(c, key=lambda s: s.offset_m)  # "how far off route" = the closest part of the stop
        merged.append(
            Stop(
                primary.poi,
                nearest.along_m,
                nearest.offset_m,
                nearest.side,
                also=[s.poi for s in c if s is not primary],
            )
        )
    merged.sort(key=lambda s: (s.along_m, s.offset_m))
    annotate_next(merged)
    return merged


def annotate_next(stops: list[Stop]) -> None:
    """Fill in distance from each stop to the next water / food stop strictly after it."""
    for resource, attr in ((WATER, "next_water_m"), (FOOD, "next_food_m")):
        pos = sorted(s.along_m for s in stops if resource in s.provides)
        for s in stops:
            k = bisect.bisect_right(pos, s.along_m + 1.0)  # +1 m: skip this stop / co-located ones
            setattr(s, attr, pos[k] - s.along_m if k < len(pos) else None)


def find_gaps(route_length_m: float, stops: list[Stop], resource: str, max_gap_m: float) -> list[Gap]:
    """Stretches longer than ``max_gap_m`` with no stop providing ``resource``.

    The route start and finish count as boundaries, so the first and last
    stretches are checked too.
    """
    pos = sorted({s.along_m for s in stops if resource in s.provides})
    marks = [0.0, *pos, route_length_m]
    gaps = []
    for k, (a, b) in enumerate(zip(marks, marks[1:])):
        if b - a > max_gap_m:
            gaps.append(Gap(resource, a, b, at_start=(k == 0), at_end=(k == len(marks) - 2)))
    return gaps
