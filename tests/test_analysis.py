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


# --- merging co-located features ---------------------------------------------------
from resupply.analysis import merge_nearby  # noqa: E402

M = 1 / 111_320  # ~1 m of latitude in degrees


def stops_for(elements, route=None):
    route = route or north_route()
    return merge_nearby(locate(route, pois_from_elements(elements, CATS), max_offset_m=500))


def test_gas_station_and_its_shop_merge():
    (s,) = stops_for([
        node(1, 40.1, -104.9995, amenity="fuel", brand="Love's"),
        node(2, 40.1 + 30 * M, -104.9995, shop="convenience", name="Loves"),
    ])
    assert s.label == "Gas station + Convenience store"
    assert s.name == "Love's" and s.poi.osm_id == 1
    assert [p.osm_id for p in s.also] == [2]
    assert s.provides == {"food", "water"}


def test_unnamed_pumps_fold_into_named_station():
    (s,) = stops_for([
        node(1, 40.1, -104.9995, amenity="fuel"),
        node(2, 40.1 + 20 * M, -104.9995, amenity="fuel", name="Arco"),
        node(3, 40.1 + 40 * M, -104.9995, amenity="fuel"),
    ])
    assert s.name == "Arco" and s.poi.osm_id == 2 and len(s.pois) == 3


def test_name_extension_merges_but_different_businesses_do_not():
    stops = stops_for([
        node(1, 40.1, -104.9995, amenity="fuel", name="Sinclair"),
        node(2, 40.1 + 25 * M, -104.9995, shop="convenience", name="Sinclair DinoMart"),
        node(3, 40.1 + 50 * M, -104.9995, shop="convenience", name="7-Eleven"),
    ])
    assert [sorted(p.osm_id for p in s.pois) for s in stops] == [[1, 2], [3]]


def test_water_never_folds_into_a_shop_but_duplicate_taps_merge():
    stops = stops_for([
        node(1, 40.1, -104.9995, shop="supermarket", name="Safeway"),
        node(2, 40.1 + 20 * M, -104.9995, amenity="drinking_water"),
        node(3, 40.1 + 30 * M, -104.9995, amenity="drinking_water"),
    ])
    assert sorted(s.label for s in stops) == ["Drinking water", "Supermarket"]
    water = next(s for s in stops if s.label == "Drinking water")
    assert len(water.pois) == 2


def test_dense_strip_does_not_chain_into_one_stop():
    # unnamed gas nodes every 50 m for 400 m: each is within 60 m of the next
    stops = stops_for([node(i, 40.1 + i * 50 * M, -104.9995, amenity="fuel") for i in range(9)])
    assert len(stops) >= 3
    assert all(len(s.pois) <= 3 for s in stops)


def test_far_apart_same_brand_stays_separate():
    stops = stops_for([
        node(1, 40.1, -104.9995, amenity="fuel", name="Shell"),
        node(2, 40.1 + 200 * M, -104.9995, amenity="fuel", name="Shell"),
    ])
    assert len(stops) == 2


def test_out_and_back_passes_stay_separate_after_merge():
    out = straight_north(40.0, 40.2)
    route = Route(*zip(*(out + out[::-1][1:])))
    stops = stops_for([
        node(1, 40.05, -104.9995, amenity="fuel", name="76"),
        node(2, 40.05 + 20 * M, -104.9995, shop="convenience", name="76"),
    ], route)
    assert len(stops) == 2 and all(len(s.pois) == 2 for s in stops)
    assert stops[1].along_m - stops[0].along_m > 10_000


def test_merge_disabled_and_next_recomputed():
    els = [
        node(1, 40.1, -104.9995, amenity="fuel", name="A"),
        node(2, 40.1 + 20 * M, -104.9995, shop="convenience", name="A"),
        node(3, 40.2, -104.9995, amenity="drinking_water"),
    ]
    raw = locate(north_route(), pois_from_elements(els, CATS), max_offset_m=500)
    assert len(merge_nearby(raw, 0)) == 3
    merged = merge_nearby(raw)
    assert len(merged) == 2
    assert merged[0].next_water_m == pytest.approx(merged[1].along_m - merged[0].along_m)


def test_per_category_offset_limits():
    # both ~850 m east of the route
    els = [
        node(1, 40.1, -104.99, amenity="drinking_water"),
        node(2, 40.2, -104.99, tourism="camp_site"),
    ]
    stops = locate(north_route(), pois_from_elements(els, CATS), {"water": 300, "campsite": 5000})
    assert [s.poi.osm_id for s in stops] == [2]


def test_narrow_category_keeps_its_own_pass_separation():
    # out-and-back 3 km spur; campsites searched to 5 km must not merge the water tap's two passes
    out = straight_north(40.0, 40.027)
    route = Route(*zip(*(out + out[::-1][1:])))
    els = [node(1, 40.005, -104.9995, amenity="drinking_water")]
    stops = locate(route, pois_from_elements(els, CATS), {"water": 300, "campsite": 5000})
    assert len(stops) == 2
