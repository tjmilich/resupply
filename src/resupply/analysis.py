"""Turn OSM elements into located stops along a route, and find resupply gaps."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
import shapely

from .categories import FOOD, WATER, Category, classify, provides
from .route import GEOD, Route

ON_ROUTE_M = 20.0  # offsets below this are reported as "on route" (no side)


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
    """One pass of the route by a POI. A POI passed twice yields two stops."""

    poi: POI
    along_m: float
    offset_m: float
    side: str  # "L", "R" or "" when on the route
    next_water_m: float | None = None
    next_food_m: float | None = None

    @property
    def provides(self) -> frozenset:
        return self.poi.provides


@dataclass
class Gap:
    resource: str
    start_m: float
    end_m: float
    at_start: bool = field(default=False)
    at_end: bool = field(default=False)

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
