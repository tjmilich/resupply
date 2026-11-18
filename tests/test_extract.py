import gzip
import json

import pytest

from resupply.extract import LocalExtract, extract_pbf, parse_poly, tile_name, write_extract

osmium = pytest.importorskip("osmium")

OSM_XML = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="test">
  <node id="1" version="1" lat="44.0" lon="-72.0">
    <tag k="amenity" v="drinking_water"/><tag k="name" v="Tap"/><tag k="wheelchair" v="yes"/>
  </node>
  <node id="2" version="1" lat="44.0" lon="-72.1"><tag k="amenity" v="bench"/></node>
  <node id="3" version="1" lat="44.0" lon="-72.2">
    <tag k="amenity" v="drinking_water"/><tag k="access" v="private"/>
  </node>
  <node id="10" version="1" lat="44.10" lon="-72.30"/>
  <node id="11" version="1" lat="44.10" lon="-72.28"/>
  <node id="12" version="1" lat="44.12" lon="-72.28"/>
  <node id="13" version="1" lat="44.12" lon="-72.30"/>
  <way id="100" version="1">
    <nd ref="10"/><nd ref="11"/><nd ref="12"/><nd ref="13"/><nd ref="10"/>
    <tag k="shop" v="supermarket"/><tag k="building" v="yes"/><tag k="name" v="Grocer"/>
  </way>
  <node id="20" version="1" lat="44.50" lon="-72.50"/>
  <node id="21" version="1" lat="44.50" lon="-72.40"/>
  <node id="22" version="1" lat="44.60" lon="-72.40"/>
  <node id="23" version="1" lat="44.60" lon="-72.50"/>
  <way id="200" version="1"><nd ref="20"/><nd ref="21"/><nd ref="22"/><nd ref="23"/><nd ref="20"/></way>
  <relation id="300" version="1">
    <member type="way" ref="200" role="outer"/>
    <tag k="type" v="multipolygon"/><tag k="tourism" v="camp_site"/><tag k="name" v="Big Camp"/>
  </relation>
</osm>
"""


@pytest.fixture
def osm_file(tmp_path):
    p = tmp_path / "t.osm"
    p.write_text(OSM_XML)
    return p


def test_extract_pbf_nodes_ways_and_multipolygons(osm_file):
    els = {(e["type"], e["id"]): e for e in extract_pbf(osm_file)}
    assert set(els) == {("node", 1), ("way", 100), ("relation", 300)}  # bench + private tap dropped
    assert els[("node", 1)]["tags"] == {"amenity": "drinking_water", "name": "Tap"}  # irrelevant tags dropped
    w = els[("way", 100)]
    assert w["lat"] == pytest.approx(44.11) and w["lon"] == pytest.approx(-72.29)  # bbox centre, like Overpass
    r = els[("relation", 300)]
    assert r["lat"] == pytest.approx(44.55, abs=1e-6) and r["lon"] == pytest.approx(-72.45, abs=1e-6)
    assert r["tags"]["tourism"] == "camp_site" and "type" not in r["tags"]


def test_parse_poly_with_hole():
    poly = parse_poly("""test
1
  -73.0  43.0
  -71.0  43.0
  -71.0  45.0
  -73.0  45.0
END
!2
  -72.2  43.8
  -71.8  43.8
  -71.8  44.2
  -72.2  44.2
END
END
""")
    assert poly.area == pytest.approx(4 - 0.16)
    assert not poly.contains(__import__("shapely").geometry.Point(-72.0, 44.0))


def test_write_and_read_extract(osm_file, tmp_path):
    cover = parse_poly("c\n1\n -73 43\n -71 43\n -71 45\n -73 45\nEND\nEND\n")
    out = tmp_path / "extract"
    m = write_extract(out, extract_pbf(osm_file), cover, ["test/region"], "2026-11-17")
    assert m["feature_count"] == 3 and m["tiles"] == {"44_-72.jsonl.gz": 1, "44_-73.jsonl.gz": 2}
    with gzip.open(out / "tiles" / "44_-73.jsonl.gz", "rt") as f:
        assert len([json.loads(x) for x in f]) == 2

    ex = LocalExtract(out)
    assert ex.data_date == "2026-11-17" and ex.regions == ["test/region"]
    line = [(44.0, -72.05), (44.2, -72.05)]
    assert ex.covers(line, 1000)
    assert not ex.covers([(44.0, -72.05), (46.0, -72.05)], 1000)  # leaves the coverage area
    near = ex.elements_near(line, 5000)  # tap is ~4 km away; grocer ~19 km
    assert [e["id"] for e in near] == [1]
    assert {e["id"] for e in ex.elements_near(line, 30_000)} == {1, 100}


def test_tile_name_negative_coords():
    assert tile_name(44.5, -72.5) == "44_-73.jsonl.gz"
    assert tile_name(-0.5, 0.5) == "-1_0.jsonl.gz"
