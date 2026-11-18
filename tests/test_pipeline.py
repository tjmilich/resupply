from shapely.geometry import box

from resupply import cli
from resupply.extract import LocalExtract, write_extract
from resupply.overpass import OverpassClient
from resupply.pipeline import Options, analyze
from resupply.route import Route

from conftest import node, straight_north


def make_extract(tmp_path, elements, north_limit):
    out = tmp_path / "extract"
    # coverage: everything south of north_limit around lon -105
    write_extract(out, elements, box(-106, 39, -104, north_limit), ["test/region"], "2026-11-01")
    return LocalExtract(out)


def test_covered_chunks_local_rest_live(tmp_path, monkeypatch):
    ex = make_extract(
        tmp_path,
        [node(1, 40.1, -104.999, amenity="drinking_water"), node(2, 40.15, -105.001, shop="supermarket")],
        north_limit=40.26,
    )
    live_queries = []

    def fake_fetch(self, q):
        live_queries.append(q)
        return {"elements": [node(3, 40.4, -104.999, amenity="drinking_water")]}

    monkeypatch.setattr(OverpassClient, "_fetch", fake_fetch)
    route = Route(*zip(*straight_north(40.0, 40.5)))  # ~55 km north; extract covers the southern part
    progress = []
    res = analyze(route, Options(chunk_km=10), OverpassClient(cache=None, min_interval_s=0), extract=ex,
                  on_progress=lambda d, n: progress.append((d, n)))
    assert res.n_chunks == 6 and 0 < res.n_local_chunks < 6
    assert len(live_queries) == res.n_chunks - res.n_local_chunks
    assert {s.poi.osm_id for s in res.stops} == {1, 2, 3}
    assert res.extract_date == "2026-11-01"
    assert progress[-1] == (6, 6) and [d for d, _ in progress] == sorted(d for d, _ in progress)


def test_fully_covered_route_makes_no_live_queries(tmp_path, monkeypatch):
    ex = make_extract(tmp_path, [node(1, 40.1, -104.999, amenity="drinking_water")], north_limit=41)

    def boom(self, q):
        raise AssertionError("should not query Overpass")

    monkeypatch.setattr(OverpassClient, "_fetch", boom)
    res = analyze(Route(*zip(*straight_north(40.0, 40.5))), Options(), OverpassClient(cache=None), extract=ex)
    assert res.n_local_chunks == res.n_chunks and [s.poi.osm_id for s in res.stops] == [1]


def test_cli_uses_extract_and_no_extract_flag(tmp_path, write_gpx, monkeypatch):
    ex_dir = make_extract(tmp_path, [node(1, 40.1, -104.999, amenity="drinking_water")], north_limit=41).dir
    calls = []
    monkeypatch.setattr(OverpassClient, "_fetch", lambda self, q: calls.append(q) or {"elements": []})
    gpx = write_gpx(straight_north())
    assert cli.main([str(gpx), "--extract", str(ex_dir), "--no-cache", "-o", str(tmp_path / "o1")]) == 0
    assert calls == []
    assert cli.main([str(gpx), "--extract", str(ex_dir), "--no-extract", "--no-cache", "-o", str(tmp_path / "o2")]) == 0
    assert calls  # went live


def test_cli_bad_extract_path_is_usage_error(tmp_path, write_gpx):
    import pytest

    with pytest.raises(SystemExit) as e:
        cli.main([str(write_gpx(straight_north())), "--extract", str(tmp_path / "nope")])
    assert e.value.code == 2
