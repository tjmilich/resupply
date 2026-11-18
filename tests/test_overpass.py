import pytest
import requests

from resupply.categories import CATEGORIES, classify, parse_category_list, provides
from resupply.overpass import (
    DiskCache,
    OverpassClient,
    OverpassError,
    SuspectResponse,
    build_query,
    fetch_elements,
    road_count,
)

from conftest import FakeResponse, FakeSession, node

OK = {"elements": [node(1, 40.1, -105.0, amenity="drinking_water")]}


def client(responses, tmp_path, **kw):
    sleeps = []
    c = OverpassClient(
        endpoints=["https://a.example/api", "https://b.example/api"],
        cache=DiskCache(tmp_path / "cache"),
        session=FakeSession(responses),
        sleep=sleeps.append,
        min_interval_s=0,
        jitter=0,
        **kw,
    )
    return c, sleeps


def test_build_query_has_around_polyline_and_all_filters():
    cats = parse_category_list(None)
    q = build_query([(40.0, -105.0), (40.1234567, -105.1)], 1050, cats)
    assert "(around:1050,40.000000,-105.000000,40.123457,-105.100000)" in q
    assert q.startswith("[out:json]")
    assert "out center tags;" in q
    assert 'way["highway"](around:150,40.000000,-105.000000,40.123457,-105.100000);\nout count;' in q
    for cat in cats:
        for f in cat.overpass_filters():
            assert f"nwr{f}(around:" in q
    assert '["amenity"="drinking_water"]["drinking_water"!="no"]' in q


def test_classify_matches_overpass_semantics():
    cats = parse_category_list(None)
    assert classify({"amenity": "drinking_water"}, cats).key == "water"
    assert classify({"amenity": "drinking_water", "drinking_water": "no"}, cats) is None
    assert classify({"amenity": "drinking_water", "access": "private"}, cats) is None
    assert classify({"amenity": "fuel", "shop": "convenience"}, cats).key == "fuel"
    assert classify({"natural": "spring"}, cats) is None
    assert classify({"shop": "bicycle"}, parse_category_list("water")) is None


def test_provides():
    fuel = CATEGORIES["fuel"]
    assert provides(fuel, {}) == {"food", "water"}
    assert provides(fuel, {}, fuel_as_food=False) == frozenset()
    assert provides(fuel, {"shop": "convenience"}, fuel_as_food=False) == {"food", "water"}
    assert provides(CATEGORIES["campsite"], {"drinking_water": "yes"}) == {"water"}
    assert provides(CATEGORIES["bike_shop"], {}) == frozenset()


def test_unknown_category():
    with pytest.raises(ValueError):
        parse_category_list("water,pubs")


def test_cache_hit_skips_http(tmp_path):
    c, _ = client([FakeResponse(200, OK)], tmp_path)
    assert c.query("Q") == (OK, False)
    assert c.query("Q") == (OK, True)
    assert len(c.session.calls) == 1


def test_refresh_bypasses_cache(tmp_path):
    c, _ = client([FakeResponse(200, OK), FakeResponse(200, {"elements": []})], tmp_path)
    c.query("Q")
    data, cached = c.query("Q", refresh=True)
    assert not cached and data == {"elements": []}
    assert c.query("Q")[0] == {"elements": []}  # cache updated


def test_cache_ttl(tmp_path):
    cache = DiskCache(tmp_path, ttl_s=-1)  # everything is expired
    cache.set("Q", OK)
    assert cache.get("Q") is None
    assert DiskCache(tmp_path).get("Q") == OK


def test_retry_rotates_endpoint_and_backs_off(tmp_path):
    c, sleeps = client([FakeResponse(429), FakeResponse(504), FakeResponse(200, OK)], tmp_path, backoff_s=2)
    data, _ = c.query("Q")
    assert data == OK
    assert [u for u, _ in c.session.calls] == ["https://a.example/api", "https://b.example/api", "https://a.example/api"]
    assert sleeps == [2, 4]


def test_runtime_error_remark_is_retried_and_not_cached(tmp_path):
    bad = {"elements": [], "remark": "runtime error: Query timed out"}
    c, _ = client([FakeResponse(200, bad), FakeResponse(200, bad)], tmp_path, max_retries=2)
    with pytest.raises(OverpassError, match="timed out"):
        c.query("Q")
    assert c.cache.get("Q") is None


def test_bad_request_fails_fast(tmp_path):
    c, _ = client([FakeResponse(400, text="parse error")], tmp_path)
    with pytest.raises(OverpassError, match="400"):
        c.query("Q")
    assert len(c.session.calls) == 1


def test_fetch_elements_dedupes_across_chunks(tmp_path):
    shared = node(1, 40.1, -105.0, amenity="drinking_water")
    resp1 = {"elements": [shared, node(2, 40.0, -105.0, shop="supermarket")]}
    resp2 = {"elements": [shared]}
    c, _ = client([FakeResponse(200, resp1), FakeResponse(200, resp2)], tmp_path)
    chunks = [[(40.0, -105.0), (40.1, -105.0)], [(40.1, -105.0), (40.2, -105.0)]]
    els, missing = fetch_elements(c, chunks, 1000, parse_category_list(None))
    assert sorted(e["id"] for e in els) == [1, 2]
    assert missing == []


def test_unresponsive_endpoint_is_skipped_for_rest_of_run(tmp_path):
    c, _ = client(
        [
            FakeResponse(502),  # a: transient
            requests.ReadTimeout("b never answers"),  # b: marked unreachable
            FakeResponse(200, OK),  # a
            FakeResponse(200, OK),  # next query goes straight to a
        ],
        tmp_path,
    )
    c.query("Q1")
    c.query("Q2")
    assert c.unreachable == {"https://b.example/api"}
    assert [u for u, _ in c.session.calls] == [
        "https://a.example/api",
        "https://b.example/api",
        "https://a.example/api",
        "https://a.example/api",
    ]


def test_all_endpoints_unreachable_still_retries(tmp_path):
    c, _ = client(
        [requests.ConnectionError("a"), requests.ConnectionError("b"), FakeResponse(200, OK)],
        tmp_path,
    )
    assert c.query("Q")[0] == OK
    assert len(c.session.calls) == 3


def test_connect_and_read_timeouts_are_separate(tmp_path):
    seen = {}

    class S:
        def post(self, url, data=None, timeout=None, headers=None):
            seen["timeout"] = timeout
            return FakeResponse(200, OK)

    c = OverpassClient(session=S(), min_interval_s=0, connect_timeout_s=7, http_timeout_s=99)
    c.query("Q")
    assert seen["timeout"] == (7, 99)


def count_el(ways):
    return {"type": "count", "id": 0, "tags": {"nodes": "0", "ways": str(ways), "relations": "0", "total": str(ways)}}


GOOD = {"elements": [node(1, 40.1, -105.0, amenity="drinking_water"), count_el(649)]}
BOGUS = {"elements": [count_el(0)]}  # what the broken backend returns


def test_road_count():
    assert road_count(GOOD) == 649
    assert road_count(BOGUS) == 0
    assert road_count(OK) is None


def test_bogus_empty_response_is_retried_and_not_cached(tmp_path):
    c, _ = client([FakeResponse(200, BOGUS), FakeResponse(200, GOOD)], tmp_path)
    data, cached = c.query("Q")
    assert data == GOOD and not cached
    assert len(c.session.calls) == 2
    assert c.query("Q") == (GOOD, True)


def test_persistently_bogus_raises_suspect_and_never_caches(tmp_path):
    c, _ = client([FakeResponse(200, BOGUS)] * 3, tmp_path, max_retries=3)
    with pytest.raises(SuspectResponse):
        c.query("Q")
    assert c.cache.get("Q") is None


def test_bogus_cache_entry_is_ignored(tmp_path):
    c, _ = client([FakeResponse(200, GOOD)], tmp_path)
    c.cache.set("Q", BOGUS)
    assert c.query("Q") == (GOOD, False)


CHUNKS2 = [[(40.0, -105.0), (40.1, -105.0)], [(40.1, -105.0), (40.2, -105.0)]]


def test_fetch_elements_reports_missing_chunks_and_strips_count(tmp_path):
    # chunk 1 ok; chunk 2 bogus in both rounds (2 attempts each)
    c, sleeps = client([FakeResponse(200, GOOD)] + [FakeResponse(200, BOGUS)] * 4, tmp_path, max_retries=2)
    els, missing = fetch_elements(c, CHUNKS2, 1000, parse_category_list(None), retry_pause_s=7)
    assert [e["id"] for e in els] == [1]
    assert missing == [1]
    assert 7 in sleeps  # paused before the second round


def test_server_error_on_one_chunk_does_not_abort_and_second_round_recovers(tmp_path):
    other = {"elements": [node(2, 40.15, -105.0, shop="supermarket"), count_el(10)]}
    c, _ = client(
        [FakeResponse(502), FakeResponse(502),  # chunk 1, round 1: exhausted
         FakeResponse(200, other),  # chunk 2
         FakeResponse(200, GOOD)],  # chunk 1, round 2
        tmp_path,
        max_retries=2,
    )
    progress = []
    els, missing = fetch_elements(c, CHUNKS2, 1000, parse_category_list(None), retry_pause_s=0,
                                  on_progress=lambda d, n: progress.append((d, n)))
    assert sorted(e["id"] for e in els) == [1, 2]
    assert missing == []
    assert progress == [(1, 2), (2, 2)]


def test_every_chunk_failing_is_an_error(tmp_path):
    c, _ = client([FakeResponse(502)] * 4, tmp_path, max_retries=1)
    with pytest.raises(OverpassError, match="server may be down"):
        fetch_elements(c, CHUNKS2, 1000, parse_category_list(None), retry_pause_s=0)


def test_build_query_per_category_radius():
    cats = parse_category_list("water,campsite")
    q = build_query([(40.0, -105.0), (40.1, -105.0)], {"water": 300, "campsite": 5000}, cats)
    assert '["amenity"="drinking_water"]["drinking_water"!="no"](around:300,' in q
    assert '["tourism"="camp_site"](around:5000,' in q
    assert "around:150," in q  # canary radius unaffected


def test_backoff_is_capped_and_jittered(tmp_path):
    c, sleeps = client([FakeResponse(502)] * 6, tmp_path, max_retries=6, backoff_s=10, max_backoff_s=30)
    with pytest.raises(OverpassError):
        c.query("Q")
    assert sleeps == [10, 20, 30, 30, 30]

    sleeps2 = []
    j = OverpassClient(endpoints=["https://a.example/api"], session=FakeSession([FakeResponse(502)] * 3),
                       sleep=sleeps2.append, min_interval_s=0, max_retries=3, backoff_s=10, jitter=0.25)
    with pytest.raises(OverpassError):
        j.query("Q")
    assert 7.5 <= sleeps2[0] <= 12.5 and 15 <= sleeps2[1] <= 25
