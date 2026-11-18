"""GPX loading, geodesic measurement, projection, simplification and chunking."""

from __future__ import annotations

from pathlib import Path

import gpxpy
import gpxpy.gpx
import numpy as np
from pyproj import CRS, Geod, Transformer
from shapely.geometry import LineString

GEOD = Geod(ellps="WGS84")

LatLon = tuple[float, float]


class RouteError(ValueError):
    pass


class Route:
    """A route as a polyline with geodesic cumulative distances and a local metric projection.

    The projection (azimuthal equidistant, centred on the route) is only used for
    nearest-segment searches and simplification. All reported distances are
    geodesic on the WGS84 ellipsoid.
    """

    def __init__(self, lat, lon, name: str = "route"):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        if lat.shape != lon.shape:
            raise RouteError("lat/lon length mismatch")
        # Drop consecutive duplicates: zero-length segments break projection maths.
        if len(lat):
            keep = np.r_[True, (np.diff(lat) != 0) | (np.diff(lon) != 0)]
            lat, lon = lat[keep], lon[keep]
        if len(lat) < 2:
            raise RouteError("route needs at least two distinct points")

        self.name = name
        self.lat = lat
        self.lon = lon
        _, _, seg = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
        self.seg_m = np.asarray(seg, dtype=float)
        self.cum_m = np.concatenate([[0.0], np.cumsum(self.seg_m)])

        lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
        self._to_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        x, y = self._to_xy.transform(lon, lat)
        self.xy = np.column_stack([x, y])

    def __len__(self) -> int:
        return len(self.lat)

    @property
    def length_m(self) -> float:
        return float(self.cum_m[-1])

    def to_xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._to_xy.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        return np.asarray(x), np.asarray(y)

    def to_latlon(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self._to_ll.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return np.asarray(lat), np.asarray(lon)

    def point_at(self, along_m: float) -> LatLon:
        """Interpolated lat/lon at a geodesic distance along the route."""
        along_m = min(max(along_m, 0.0), self.length_m)
        i = int(np.searchsorted(self.cum_m, along_m, side="right") - 1)
        i = min(i, len(self.seg_m) - 1)
        t = (along_m - self.cum_m[i]) / self.seg_m[i]
        x, y = self.xy[i] + t * (self.xy[i + 1] - self.xy[i])
        lat, lon = self.to_latlon(x, y)
        return float(lat), float(lon)

    def slice_coords(self, start_m: float, end_m: float, tolerance_m: float = 0.0) -> list[LatLon]:
        """Lat/lon coordinates of the route between two along-route distances."""
        start_m, end_m = max(start_m, 0.0), min(end_m, self.length_m)
        inner = np.nonzero((self.cum_m > start_m) & (self.cum_m < end_m))[0]
        coords = [self.point_at(start_m)]
        coords += list(zip(self.lat[inner].tolist(), self.lon[inner].tolist()))
        coords.append(self.point_at(end_m))
        if tolerance_m > 0 and len(coords) > 2:
            coords = _simplify_latlon(self, coords, tolerance_m)
        return coords


def load_gpx(path: str | Path) -> Route:
    """Load all track segments (or, failing that, routes) from a GPX file, in order."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
    except OSError as e:
        raise RouteError(f"cannot read {path}: {e}") from e
    except gpxpy.gpx.GPXException as e:
        raise RouteError(f"invalid GPX in {path}: {e}") from e

    pts: list[LatLon] = []
    for trk in gpx.tracks:
        for seg in trk.segments:
            pts.extend((p.latitude, p.longitude) for p in seg.points)
    if not pts:
        for rte in gpx.routes:
            pts.extend((p.latitude, p.longitude) for p in rte.points)
    if not pts:
        raise RouteError(f"{path} contains no track or route points")

    name = gpx.name or next((t.name for t in gpx.tracks if t.name), None) or next(
        (r.name for r in gpx.routes if r.name), None
    ) or path.stem
    lat, lon = zip(*pts)
    return Route(lat, lon, name=name)


def _simplify_latlon(route: Route, coords: list[LatLon], tolerance_m: float) -> list[LatLon]:
    lat, lon = zip(*coords)
    x, y = route.to_xy(lat, lon)
    simp = LineString(np.column_stack([x, y])).simplify(tolerance_m, preserve_topology=False)
    sx, sy = np.asarray(simp.coords).T
    slat, slon = route.to_latlon(sx, sy)
    return list(zip(slat.tolist(), slon.tolist()))


def simplified_coords(route: Route, tolerance_m: float) -> list[LatLon]:
    """Douglas–Peucker simplification in metres, returned as (lat, lon)."""
    coords = list(zip(route.lat.tolist(), route.lon.tolist()))
    if tolerance_m <= 0:
        return coords
    return _simplify_latlon(route, coords, tolerance_m)


def _densify(coords: list[LatLon], max_seg_m: float) -> list[LatLon]:
    """Insert geodesic intermediate points so no segment exceeds ``max_seg_m``.

    Heavy simplification can leave a single long straight segment; without this
    it would land in one chunk and defeat the per-query length cap.
    """
    out = [coords[0]]
    for (lat1, lon1), (lat2, lon2) in zip(coords, coords[1:]):
        _, _, d = GEOD.inv(lon1, lat1, lon2, lat2)
        n = int(np.ceil(d / max_seg_m)) - 1
        if n > 0:
            out.extend((la, lo) for lo, la in GEOD.npts(lon1, lat1, lon2, lat2, n))
        out.append((lat2, lon2))
    return out


def chunk_coords(coords: list[LatLon], max_points: int = 150, max_length_m: float = 50_000) -> list[list[LatLon]]:
    """Split a polyline into consecutive chunks that share their boundary vertex.

    A chunk closes when it reaches ``max_points`` vertices or adding the next
    segment would exceed ``max_length_m``. Sharing the boundary vertex means the
    union of per-chunk ``around`` corridors covers the whole route.
    """
    if max_points < 2:
        raise ValueError("max_points must be >= 2")
    if len(coords) < 2:
        return [list(coords)] if coords else []
    coords = _densify(coords, max_length_m)
    lats = np.array([c[0] for c in coords])
    lons = np.array([c[1] for c in coords])
    _, _, seg = GEOD.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])

    chunks: list[list[LatLon]] = []
    cur = [coords[0]]
    cur_len = 0.0
    for i in range(1, len(coords)):
        d = float(seg[i - 1])
        if len(cur) >= max_points or (len(cur) >= 2 and cur_len + d > max_length_m):
            chunks.append(cur)
            cur = [coords[i - 1]]
            cur_len = 0.0
        cur.append(coords[i])
        cur_len += d
    chunks.append(cur)
    return chunks


def chunk_ranges(chunks: list[list[LatLon]], route_length_m: float) -> list[tuple[float, float]]:
    """Approximate along-route (start, end) distances of each chunk.

    Chunk lengths are measured on the simplified polyline and scaled to the true
    route length, which is accurate enough for reporting.
    """
    lengths = []
    for c in chunks:
        lat = np.array([p[0] for p in c])
        lon = np.array([p[1] for p in c])
        lengths.append(float(np.sum(GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2])) if len(c) > 1 else 0.0)
    total = sum(lengths) or 1.0
    scale = route_length_m / total
    out, pos = [], 0.0
    for L in lengths:
        out.append((pos, min(pos + L * scale, route_length_m)))
        pos += L * scale
    return out
