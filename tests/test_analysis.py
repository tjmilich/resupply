import pytest

from resupply.analysis import find_gaps, locate, pois_from_elements
from resupply.categories import parse_category_list
from resupply.route import GEOD, Route

from conftest import node, straight_north

CATS = parse_category_list(None)


def north_route():
    return Route(*zip(*straight_north()))


def test_along_offset_and_side():
    route = north_route()
    pois = pois_from_elements(
        [
            node(1, 40.1, -104.995, amenity="drinking_water"),  # east of a northbound route -> right
            node(2, 40.3, -105.004, shop="supermarket"),  # west -> left
        ],
        CATS,
    )
    stops = locate(route, pois, max_offset_m=1000)
    assert [s.poi.osm_id for s in stops] == [1, 2]
    s1, s2 = stops

    _, _, along1 = GEOD.inv(-105.0, 40.0, -105.0, 40.1)
    assert s1.along_m == pytest.approx(along1, abs=2)
    _, _, off1 = GEOD.inv(-104.995, 40.1, -105.0, 40.1)
    assert s1.offset_m == pytest.approx(off1, abs=1)
    assert s1.side == "R" and s2.side == "L"

    # supermarkets count as both food and water
    assert s1.next_water_m == pytest.approx(s2.along_m - s1.along_m)
    assert s1.next_food_m == pytest.approx(s2.along_m - s1.along_m)
    assert s2.next_food_m is None and s2.next_water_m is None


def test_points_beyond_radius_dropped():
    route = north_route()
    pois = pois_from_elements([node(1, 40.1, -104.98, amenity="drinking_water")], CATS)  # ~1.7 km off
    assert locate(route, pois, max_offset_m=1000) == []
    assert len(locate(route, pois, max_offset_m=2000)) == 1


def test_beyond_ends_measures_to_endpoint():
    route = north_route()
    pois = pois_from_elements([node(1, 40.505, -105.0, amenity="drinking_water")], CATS)
    (s,) = locate(route, pois, max_offset_m=1000)
    assert s.along_m == pytest.approx(route.length_m)
    assert s.offset_m == pytest.approx(555, abs=5)


def test_out_and_back_gives_two_passes():
    out = straight_north(40.0, 40.2)
    route = Route(*zip(*(out + out[::-1][1:])))
    pois = pois_from_elements([node(1, 40.05, -104.999, amenity="drinking_water")], CATS)
    stops = locate(route, pois, max_offset_m=500)
    assert len(stops) == 2
    half = route.length_m / 2
    assert stops[0].along_m == pytest.approx(half * 0.25, rel=0.01)
    assert stops[1].along_m == pytest.approx(half * 1.75, rel=0.01)
    assert stops[0].side == "R" and stops[1].side == "L"


def test_turnaround_point_is_single_pass():
    out = straight_north(40.0, 40.2)
    route = Route(*zip(*(out + out[::-1][1:])))
    pois = pois_from_elements([node(1, 40.2005, -105.0, amenity="drinking_water")], CATS)
    assert len(locate(route, pois, max_offset_m=500)) == 1


def test_way_center_used():
    els = [{"type": "way", "id": 9, "center": {"lat": 40.2, "lon": -105.001}, "tags": {"tourism": "camp_site"}}]
    (p,) = pois_from_elements(els, CATS)
    assert (p.lat, p.lon, p.osm_url) == (40.2, -105.001, "https://www.openstreetmap.org/way/9")


def test_gaps():
    route = north_route()  # ~55.5 km
    pois = pois_from_elements(
        [
            node(1, 40.1, -105.0, amenity="drinking_water"),  # ~11 km, water only
            node(2, 40.15, -105.0, shop="bicycle"),  # provides nothing
            node(3, 40.3, -105.0, shop="convenience"),  # ~33 km, food + water
        ],
        CATS,
    )
    stops = locate(route, pois, max_offset_m=500)
    water = find_gaps(route.length_m, stops, "water", 20_000)
    # bike shop provides nothing; the convenience store counts as water
    assert [(round(g.start_m / 1000), round(g.end_m / 1000)) for g in water] == [(11, 33), (33, 56)]
    assert water[-1].at_end and not water[0].at_start

    food = find_gaps(route.length_m, stops, "food", 20_000)
    assert [(round(g.start_m / 1000), round(g.end_m / 1000)) for g in food] == [(0, 33), (33, 56)]
    assert food[0].at_start

    assert find_gaps(route.length_m, [], "water", 100_000) == []
    (whole,) = find_gaps(route.length_m, [], "water", 10_000)
    assert whole.at_start and whole.at_end
