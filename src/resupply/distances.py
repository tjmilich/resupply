"""Parsing human-friendly distances and per-category search radii."""

from __future__ import annotations

import re
from typing import Iterable

from .categories import CATEGORIES, Category

_UNITS_M = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0, "kilometre": 1000.0, "kilometres": 1000.0,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
}
_DIST_RE = re.compile(r"^\s*(\d+(?:\.\d*)?|\.\d+)\s*([a-z]*)\s*$", re.IGNORECASE)

DEFAULT_RADIUS_M = 1000.0


def parse_distance(text: str) -> float:
    """Parse "500", "500m", "1.5 km", "0.5mi", "800ft" into metres. A bare number means metres."""
    m = _DIST_RE.match(text)
    if not m:
        raise ValueError(f"can't read distance {text!r} (examples: 500m, 1.5km, 0.5mi, 800ft)")
    value, unit = float(m.group(1)), m.group(2).lower()
    if unit and unit not in _UNITS_M:
        raise ValueError(f"unknown unit {m.group(2)!r} in {text!r} (use m, km, mi or ft)")
    metres = value * _UNITS_M.get(unit, 1.0)
    if metres <= 0:
        raise ValueError(f"distance must be positive: {text!r}")
    return metres


def parse_radius_spec(
    spec: str, categories: Iterable[Category], default_m: float = DEFAULT_RADIUS_M
) -> dict[str, float]:
    """Parse a --radius value into {category key: metres} for the selected categories.

    Forms (comma-separated, any order):
      ``1km``                      one radius for everything
      ``water=300m,campsite=5km``  overrides; other categories keep the default
      ``2km,water=300m``           a bare item sets the default for everything not listed
    """
    cats = list(categories)
    default = default_m
    overrides: dict[str, float] = {}
    saw_default = False
    for item in (p.strip() for p in spec.split(",")):
        if not item:
            continue
        if "=" in item:
            key, _, value = (s.strip() for s in item.partition("="))
            key = key.lower().replace("-", "_")
            if key not in CATEGORIES:
                raise ValueError(f"unknown category {key!r} in --radius (choose from {', '.join(CATEGORIES)})")
            if key in overrides:
                raise ValueError(f"category {key!r} given twice in --radius")
            overrides[key] = parse_distance(value)
        else:
            if saw_default:
                raise ValueError("--radius can have only one default distance (the item without '=')")
            default, saw_default = parse_distance(item), True
    return {c.key: overrides.get(c.key, default) for c in cats}


def describe_radii(radii: dict[str, float], short_unit: str = "m", short_m: float = 1.0) -> str:
    """Compact human summary, e.g. "1000 m (water 300 m, campsite 5000 m)"."""
    if not radii:
        return ""
    values = list(radii.values())
    common = max(set(values), key=values.count)

    def fmt(m: float) -> str:
        return f"{m / short_m:.0f} {short_unit}"

    others = [f"{CATEGORIES[k].label.lower()} {fmt(v)}" for k, v in radii.items() if v != common]
    return fmt(common) + (f" ({', '.join(others)})" if others else "")
