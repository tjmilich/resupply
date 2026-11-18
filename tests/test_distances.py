import pytest

from resupply.categories import parse_category_list
from resupply.distances import describe_radii, parse_distance, parse_radius_spec

ALL = parse_category_list(None)


@pytest.mark.parametrize(
    "text, metres",
    [
        ("1000", 1000),
        ("500m", 500),
        ("1.5km", 1500),
        ("1.5 KM", 1500),
        ("0.5mi", 804.672),
        ("2 miles", 3218.688),
        ("800ft", 243.84),
        ("1 foot", 0.3048),
        (".5km", 500),
        ("3 metres", 3),
    ],
)
def test_parse_distance(text, metres):
    assert parse_distance(text) == pytest.approx(metres)


@pytest.mark.parametrize("text", ["", "km", "-5m", "0", "5 parsecs", "1,000m", "1.2.3km"])
def test_parse_distance_rejects(text):
    with pytest.raises(ValueError):
        parse_distance(text)


def test_single_radius_applies_to_all():
    r = parse_radius_spec("0.5mi", ALL)
    assert set(r) == {c.key for c in ALL}
    assert all(v == pytest.approx(804.672) for v in r.values())


def test_overrides_keep_default_for_the_rest():
    r = parse_radius_spec("water=300m, campsite=5km", ALL)
    assert r["water"] == 300 and r["campsite"] == 5000
    assert r["supermarket"] == 1000  # built-in default


def test_bare_item_sets_default_in_any_position():
    assert parse_radius_spec("water=300m,2km", ALL)["fuel"] == 2000
    r = parse_radius_spec("2km,bike-shop=10km", ALL)
    assert r["bike_shop"] == 10_000 and r["water"] == 2000


def test_only_selected_categories_returned():
    r = parse_radius_spec("water=300m,campsite=5km", parse_category_list("water,supermarket"))
    assert r == {"water": 300, "supermarket": 1000}


@pytest.mark.parametrize(
    "spec, msg",
    [
        ("pubs=1km", "unknown category"),
        ("water=300m,water=400m", "twice"),
        ("1km,2km", "only one default"),
        ("water=lots", "can't read"),
    ],
)
def test_radius_spec_errors(spec, msg):
    with pytest.raises(ValueError, match=msg):
        parse_radius_spec(spec, ALL)


def test_describe_radii():
    r = parse_radius_spec("water=300m", ALL)
    assert describe_radii(r) == "1000 m (drinking water 300 m)"
    assert describe_radii(parse_radius_spec("1km", ALL), "ft", 0.3048) == "3281 ft"
