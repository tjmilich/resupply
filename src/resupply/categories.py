"""Resupply categories.

Each category is defined once as structured tag conditions. The same definition
generates the Overpass filter strings *and* classifies returned elements in
Python, so the two can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# (key, op, value) where op is "=" or "!=". Overpass semantics: ["k"!="v"]
# also matches elements that lack the key entirely, which tags.get() mirrors.
Cond = tuple[str, str, str]

WATER = "water"
FOOD = "food"


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    selectors: tuple[tuple[Cond, ...], ...]  # OR of ANDs
    color: str  # folium.Icon colour
    icon: str  # Font Awesome 4 icon name

    def overpass_filters(self) -> list[str]:
        return ["".join(f'["{k}"{op}"{v}"]' for k, op, v in sel) for sel in self.selectors]

    def matches(self, tags: Mapping[str, str]) -> bool:
        return any(all(_check(tags, c) for c in sel) for sel in self.selectors)


def _check(tags: Mapping[str, str], cond: Cond) -> bool:
    key, op, value = cond
    if op == "=":
        return tags.get(key) == value
    if op == "!=":
        return tags.get(key) != value
    raise ValueError(f"unsupported operator {op!r}")


# Order matters: an element matching several categories gets the first one
# (e.g. a fuel station tagged shop=convenience is reported as a gas station).
CATEGORIES: dict[str, Category] = {
    c.key: c
    for c in (
        Category(
            "water",
            "Drinking water",
            (
                (("amenity", "=", "drinking_water"), ("drinking_water", "!=", "no")),
                (("amenity", "=", "water_point"), ("drinking_water", "!=", "no")),
                (("man_made", "=", "water_tap"), ("drinking_water", "=", "yes")),
                (("natural", "=", "spring"), ("drinking_water", "=", "yes")),
                (("amenity", "=", "fountain"), ("drinking_water", "=", "yes")),
            ),
            "blue",
            "tint",
        ),
        Category("supermarket", "Supermarket", ((("shop", "=", "supermarket"),),), "green", "shopping-cart"),
        Category("fuel", "Gas station", ((("amenity", "=", "fuel"),),), "orange", "car"),
        Category("convenience", "Convenience store", ((("shop", "=", "convenience"),),), "darkgreen", "shopping-basket"),
        Category("campsite", "Campsite", ((("tourism", "=", "camp_site"),),), "darkpurple", "fire"),
        Category("bike_shop", "Bike shop", ((("shop", "=", "bicycle"),),), "cadetblue", "bicycle"),
    )
}

_NO_ACCESS = {"private", "no"}


def parse_category_list(spec: str | None) -> list[Category]:
    """Parse a comma-separated list of category keys (None/'all' = everything)."""
    if not spec or spec.strip().lower() == "all":
        return list(CATEGORIES.values())
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in CATEGORIES]
    if unknown:
        raise ValueError(f"unknown categories: {', '.join(unknown)} (choose from {', '.join(CATEGORIES)})")
    # keep canonical priority order
    return [c for k, c in CATEGORIES.items() if k in keys]


def classify(tags: Mapping[str, str], categories: list[Category]) -> Category | None:
    """Return the first category that matches, or None. Private/no-access features are skipped."""
    if tags.get("access") in _NO_ACCESS:
        return None
    for cat in categories:
        if cat.matches(tags):
            return cat
    return None


def provides(category: Category, tags: Mapping[str, str], fuel_as_food: bool = True) -> frozenset[str]:
    """Which resources (water/food) a feature can be relied on for, for gap analysis."""
    if category.key == "water":
        return frozenset({WATER})
    if category.key in ("supermarket", "convenience"):
        return frozenset({FOOD, WATER})
    if category.key == "fuel":
        has_shop = tags.get("shop") not in (None, "no")
        return frozenset({FOOD, WATER}) if (fuel_as_food or has_shop) else frozenset()
    if category.key == "campsite":
        return frozenset({WATER}) if tags.get("drinking_water") == "yes" else frozenset()
    return frozenset()
