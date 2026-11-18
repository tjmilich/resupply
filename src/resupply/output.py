"""CSV cue sheet and folium map output."""

from __future__ import annotations

import csv
import html
import io
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import folium

from .analysis import Gap, Stop
from .categories import CATEGORIES, FOOD, WATER
from .route import Route, simplified_coords


@dataclass(frozen=True)
class Units:
    long: str
    long_m: float
    short: str
    short_m: float

    def dist(self, m: float | None, nd: int = 1) -> str:
        return "" if m is None else f"{m / self.long_m:.{nd}f}"

    def off(self, m: float) -> str:
        return f"{m / self.short_m:.0f}"


KM = Units("km", 1000.0, "m", 1.0)
MI = Units("mi", 1609.344, "ft", 0.3048)

_GAP_STYLE = {WATER: ("#d62728", None), FOOD: ("#ff7f0e", "10 8")}


def _notes(stop: Stop) -> str:
    notes = []
    if any(p.tags.get("fee") == "yes" for p in stop.pois):
        notes.append("fee")
    seasonal = stop.tag("seasonal")
    if seasonal and seasonal != "no":
        notes.append(f"seasonal: {seasonal}")
    for p in stop.pois:
        if p.category.key == "campsite" and "drinking_water" in p.tags:
            notes.append(f"drinking_water={p.tags['drinking_water']}")
        if p.category.key == "fuel" and p.tags.get("shop"):
            notes.append(f"shop={p.tags['shop']}")
    if stop.tag("phone"):
        notes.append(f"phone {stop.tag('phone')}")
    if stop.also:
        notes.append(f"{len(stop.pois)} OSM features merged")
    return "; ".join(dict.fromkeys(notes))


def gap_message(gap: Gap, units: Units) -> str:
    until = "the finish" if gap.at_end else f"{units.long} {units.dist(gap.end_m)}"
    msg = f"No {gap.resource} for {units.dist(gap.length_m)} {units.long} (until {until})"
    if gap.uncertain:
        msg += " - includes a stretch with no map data, may be wrong"
    return msg


def missing_message(start_m: float, end_m: float, units: Units) -> str:
    return (
        f"NO MAP DATA {units.long} {units.dist(start_m)}-{units.dist(end_m)}: the Overpass server "
        "returned nothing usable; resupply here is unknown. Re-run later with --refresh"
    )


def write_csv(
    path: str | Path | IO[str],
    stops: list[Stop],
    gaps: list[Gap],
    units: Units = KM,
    missing: list[tuple[float, float]] = (),
) -> None:
    """Cue sheet sorted by distance, with gap warnings inlined at the start of each gap."""
    header = [
        f"dist_{units.long}",
        f"offset_{units.short}",
        "side",
        "category",
        "name",
        "provides",
        f"next_water_{units.long}",
        f"next_food_{units.long}",
        "opening_hours",
        "lat",
        "lon",
        "osm_url",
        "notes",
    ]
    rows: list[tuple[float, int, list]] = []
    for s in stops:
        p = s.poi
        rows.append(
            (
                s.along_m,
                0,
                [
                    units.dist(s.along_m, 2),
                    units.off(s.offset_m),
                    s.side,
                    s.label,
                    s.name,
                    "+".join(sorted(s.provides)),
                    units.dist(s.next_water_m),
                    units.dist(s.next_food_m),
                    s.tag("opening_hours"),
                    f"{p.lat:.6f}",
                    f"{p.lon:.6f}",
                    p.osm_url,
                    _notes(s),
                ],
            )
        )
    for g in gaps:
        row = [""] * len(header)
        row[0] = units.dist(g.start_m, 2)
        row[3] = f"GAP ({g.resource})"
        row[4] = gap_message(g, units)
        rows.append((g.start_m, 1, row))
    for a, b in missing:
        row = [""] * len(header)
        row[0] = units.dist(a, 2)
        row[3] = "NO DATA"
        row[4] = missing_message(a, b, units)
        rows.append((a, -1, row))
    rows.sort(key=lambda r: (r[0], r[1]))
    if hasattr(path, "write"):
        w = csv.writer(path)
        w.writerow(header)
        w.writerows(r[2] for r in rows)
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(r[2] for r in rows)


def csv_text(stops: list[Stop], gaps: list[Gap], units: Units = KM, missing: list[tuple[float, float]] = ()) -> str:
    buf = io.StringIO()
    write_csv(buf, stops, gaps, units, missing)
    return buf.getvalue()


def _nice_interval(length: float) -> float:
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500):
        if length / step <= 40:
            return step
    return 1000


def _popup(stops: list[Stop], units: Units) -> str:
    s0 = stops[0]
    p = s0.poi
    e = html.escape
    title = e(s0.name) or e(s0.label)
    dists = ", ".join(f"{units.long} {units.dist(s.along_m)}" for s in stops)
    side = {"L": " left", "R": " right", "": ""}[s0.side]
    parts = [
        f"<b>{title}</b>",
        e(s0.label),
        f"At {dists}",
        f"{units.off(s0.offset_m)} {units.short} off route{side}",
    ]
    if s0.tag("opening_hours"):
        parts.append(f"Hours: {e(s0.tag('opening_hours'))}")
    if s0.next_water_m is not None:
        parts.append(f"Next water: {units.dist(s0.next_water_m)} {units.long}")
    if s0.next_food_m is not None:
        parts.append(f"Next food: {units.dist(s0.next_food_m)} {units.long}")
    if _notes(s0):
        parts.append(e(_notes(s0)))
    parts.append(f'<a href="{p.osm_url}" target="_blank" rel="noopener">View on OpenStreetMap</a>')
    return "<br>".join(parts)


def write_map(
    path: str | Path,
    route: Route,
    stops: list[Stop],
    gaps: list[Gap],
    units: Units = KM,
    missing: list[tuple[float, float]] = (),
) -> None:
    build_map(route, stops, gaps, units, missing).save(str(path))


def map_html(route: Route, stops: list[Stop], gaps: list[Gap], units: Units = KM, missing=()) -> str:
    return build_map(route, stops, gaps, units, missing).get_root().render()


def build_map(
    route: Route,
    stops: list[Stop],
    gaps: list[Gap],
    units: Units = KM,
    missing: list[tuple[float, float]] = (),
) -> folium.Map:
    lats, lons = route.lat, route.lon
    bounds = [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]]
    m = folium.Map(location=[float(lats.mean()), float(lons.mean())], zoom_start=10, tiles=None, control_scale=True)
    # CyclOSM is the default: tile.openstreetmap.org rejects requests without a Referer,
    # which is exactly what a browser sends for a map opened as a local file.
    folium.TileLayer(
        tiles="https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png",
        attr='<a href="https://www.cyclosm.org">CyclOSM</a> | Map data &copy; '
        '<a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="CyclOSM",
        max_zoom=20,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr='Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, '
        'SRTM | Style &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)',
        name="OpenTopoMap",
        max_zoom=17,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap (blank if opened as a file)",
        show=False,
    ).add_to(m)

    # Route (lightly simplified so huge GPX files don't bloat the HTML)
    route_fg = folium.FeatureGroup(name="Route")
    folium.PolyLine(simplified_coords(route, 5.0), color="#3366cc", weight=4, opacity=0.85,
                    tooltip=html.escape(route.name)).add_to(route_fg)
    folium.Marker([float(lats[0]), float(lons[0])], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(route_fg)
    folium.Marker([float(lats[-1]), float(lons[-1])], tooltip="Finish",
                  icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")).add_to(route_fg)
    route_fg.add_to(m)

    # Distance markers
    total = route.length_m / units.long_m
    step = _nice_interval(total)
    dist_fg = folium.FeatureGroup(name=f"Distance markers ({step:g} {units.long})")
    d = step
    while d < total:
        lat, lon = route.point_at(d * units.long_m)
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                icon_size=(36, 16), icon_anchor=(18, 8),
                html=f'<div style="font:bold 10px sans-serif;background:#fff;border:1px solid #3366cc;'
                f'border-radius:8px;text-align:center;color:#3366cc;">{d:g}</div>',
            ),
        ).add_to(dist_fg)
        d += step
    dist_fg.add_to(m)

    # Gaps
    if gaps:
        gap_fg = folium.FeatureGroup(name=f"Gaps ({len(gaps)})")
        for g in gaps:
            color, dash = _GAP_STYLE.get(g.resource, ("#d62728", None))
            g_coords = route.slice_coords(g.start_m, g.end_m, tolerance_m=5.0)
            folium.PolyLine(
                g_coords,
                color=color, weight=10, opacity=0.55, dash_array=dash,
                tooltip=html.escape(gap_message(g, units)),
            ).add_to(gap_fg)
            folium.CircleMarker(g_coords[0], radius=5, color=color, fill=True, fill_opacity=1,
                                tooltip=html.escape(gap_message(g, units))).add_to(gap_fg)
        gap_fg.add_to(m)

    if missing:
        miss_fg = folium.FeatureGroup(name=f"No map data ({len(missing)})")
        for a, b in missing:
            folium.PolyLine(
                route.slice_coords(a, b, tolerance_m=5.0),
                color="#555555", weight=12, opacity=0.6, dash_array="4 10",
                tooltip=html.escape(missing_message(a, b, units)),
            ).add_to(miss_fg)
        miss_fg.add_to(m)

    # One marker per stop (keyed by its primary OSM feature), listing every pass in the popup.
    # A merged stop sits in its primary category's layer.
    by_poi: dict[tuple[str, int], list[Stop]] = {}
    for s in stops:
        by_poi.setdefault((s.poi.osm_type, s.poi.osm_id), []).append(s)
    for cat in CATEGORIES.values():
        groups = [g for g in by_poi.values() if g[0].poi.category.key == cat.key]
        if not groups:
            continue
        fg = folium.FeatureGroup(name=f"{cat.label} ({len(groups)})")
        for g in groups:
            p = g[0].poi
            folium.Marker(
                [p.lat, p.lon],
                tooltip=html.escape(g[0].name or g[0].label),
                popup=folium.Popup(_popup(g, units), max_width=320),
                icon=folium.Icon(color=cat.color, icon=cat.icon, prefix="fa"),
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)  # expanded, it covers the map on phones
    m.fit_bounds(bounds)

    summary = (
        f"<b>{html.escape(route.name)}</b><br>{units.dist(route.length_m)} {units.long} &middot; "
        f"{len(by_poi)} resupply points"
        + (f"<br><span style='color:#d62728'>{len(gaps)} gap(s) flagged</span>" if gaps else "")
        + (f"<br><span style='color:#555'><b>{len(missing)} stretch(es) with no map data</b></span>" if missing else "")
    )
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:rgba(255,255,255,.92);'
        f'padding:8px 10px;border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3);">'
        f"{summary}</div>"
    ))
    return m
