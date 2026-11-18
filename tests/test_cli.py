import csv

from resupply import cli
from resupply.overpass import OverpassClient

from conftest import node, straight_north

ELEMENTS = [
    node(1, 40.1, -104.995, amenity="drinking_water", name="Trailhead tap"),
    node(2, 40.12, -105.003, shop="supermarket", name="Grocer & Co <b>", opening_hours="Mo-Su 07:00-22:00"),
    node(3, 40.4, -105.0, tourism="camp_site", drinking_water="yes", fee="yes"),
    node(4, 40.2, -105.002, shop="bicycle"),
    node(5, 40.2, -104.9, shop="convenience"),  # ~8.5 km away: outside radius
]


def test_end_to_end(write_gpx, tmp_path, monkeypatch):
    calls = []

    def fake_fetch(self, q):
        calls.append(q)
        return {"elements": ELEMENTS}

    monkeypatch.setattr(OverpassClient, "_fetch", fake_fetch)
    gpx = write_gpx(straight_north(), name="E2E")
    out = tmp_path / "out"
    argv = [str(gpx), "-o", str(out), "--cache-dir", str(tmp_path / "cache"), "--max-gap", "20", "--chunk-km", "20"]

    assert cli.main(argv) == 0
    n_queries = len(calls)
    assert n_queries == 3  # ~55 km route, 20 km chunks
    assert "around:1050," in calls[0]  # radius + simplify tolerance

    with (out / "route_cues.csv").open() as f:
        rows = list(csv.DictReader(f))
    cats = [r["category"] for r in rows]
    # water at ~11 km, supermarket ~13 km, bike ~22 km, campsite (water) ~44 km, finish ~55.5 km
    assert cats == ["Drinking water", "Supermarket", "GAP (food)", "GAP (water)", "Bike shop", "Campsite"]
    tap = rows[0]
    assert tap["side"] == "R" and tap["provides"] == "water" and float(tap["offset_m"]) > 400
    expected_next_food = float(rows[1]["dist_km"]) - float(tap["dist_km"])
    assert abs(float(tap["next_food_km"]) - expected_next_food) < 0.06
    assert "No food" in rows[2]["name"] and "the finish" in rows[2]["name"]
    assert "No water" in rows[3]["name"] and "until km 44" in rows[3]["name"]
    assert "fee" in rows[5]["notes"]

    html = (out / "route_resupply.html").read_text()
    assert "leaflet" in html.lower()
    assert "Grocer &amp; Co &lt;b&gt;" in html  # names are escaped
    assert "Gaps (2)" in html

    # second run is served entirely from cache
    assert cli.main(argv) == 0
    assert len(calls) == n_queries


def test_miles_and_custom_paths(write_gpx, tmp_path, monkeypatch):
    monkeypatch.setattr(OverpassClient, "_fetch", lambda self, q: {"elements": ELEMENTS})
    gpx = write_gpx(straight_north())
    m, c = tmp_path / "m.html", tmp_path / "c.csv"
    assert cli.main([str(gpx), "--map", str(m), "--csv", str(c), "--units", "mi", "--no-cache", "-q"]) == 0
    header = c.read_text().splitlines()[0]
    assert header.startswith("dist_mi,offset_ft,")
    assert m.exists()


def test_missing_file_returns_error(tmp_path):
    assert cli.main([str(tmp_path / "nope.gpx"), "--no-cache"]) == 1


def test_missing_chunk_flags_no_data_and_exit_code(write_gpx, tmp_path, monkeypatch):
    from resupply.overpass import SuspectResponse

    import resupply.overpass as ov

    monkeypatch.setattr(ov, "RETRY_ROUND_PAUSE_S", 0)
    calls = []
    bad = {}

    def fake_fetch(self, q):
        calls.append(q)
        if len(calls) == 2:
            bad["q"] = q  # middle chunk: server keeps returning nothing, in both rounds
        if q == bad.get("q"):
            raise SuspectResponse("no data")
        return {"elements": ELEMENTS}

    monkeypatch.setattr(OverpassClient, "_fetch", fake_fetch)
    gpx = write_gpx(straight_north())
    out = tmp_path / "out"
    argv = [str(gpx), "-o", str(out), "--no-cache", "--max-gap", "20", "--chunk-km", "20"]
    assert cli.main(argv) == cli.EXIT_INCOMPLETE

    with (out / "route_cues.csv").open() as f:
        rows = list(csv.DictReader(f))
    nodata = [r for r in rows if r["category"] == "NO DATA"]
    assert len(nodata) == 1
    start = float(nodata[0]["dist_km"])
    assert 15 < start < 25  # second of three ~18.5 km chunks
    gaps = [r for r in rows if r["category"].startswith("GAP")]
    assert gaps and all("may be wrong" in r["name"] for r in gaps)  # both gaps cross the hole
    assert "No map data (1)" in (out / "route_resupply.html").read_text()


def test_radius_spec_with_units_and_categories(write_gpx, tmp_path, monkeypatch):
    calls = []

    def fake_fetch(self, q):
        calls.append(q)
        return {"elements": ELEMENTS}

    monkeypatch.setattr(OverpassClient, "_fetch", fake_fetch)
    gpx = write_gpx(straight_north())
    out = tmp_path / "out"
    # tap is ~430 m off route: excluded by water=0.2mi (~322 m); supermarket ~255 m stays
    argv = [str(gpx), "-o", str(out), "--no-cache", "--units", "mi", "-r", "1mi,water=0.2mi", "--chunk-km", "100"]
    assert cli.main(argv) == 0
    assert '["amenity"="drinking_water"]["drinking_water"!="no"](around:372,' in calls[0]  # 322 + 50 simplify
    assert '["shop"="supermarket"](around:1659,' in calls[0]
    cats = [r["category"] for r in csv.DictReader((out / "route_cues.csv").open())]
    assert "Drinking water" not in cats and "Supermarket" in cats


def test_bad_radius_is_a_usage_error(write_gpx, capsys):
    import pytest

    with pytest.raises(SystemExit) as e:
        cli.main([str(write_gpx(straight_north())), "-r", "water=far"])
    assert e.value.code == 2
    assert "can't read distance" in capsys.readouterr().err
