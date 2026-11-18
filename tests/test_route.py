import numpy as np
import pytest

from resupply.route import GEOD, Route, RouteError, chunk_coords, load_gpx, simplified_coords

from conftest import straight_north


def test_load_gpx_length_matches_geodesic(write_gpx):
    route = load_gpx(write_gpx(straight_north(), name="North"))
    assert route.name == "North"
    _, _, expected = GEOD.inv(-105.0, 40.0, -105.0, 40.5)
    assert route.length_m == pytest.approx(expected, rel=1e-6)


def test_duplicate_points_dropped():
    r = Route([40, 40, 40.1, 40.1, 40.2], [-105] * 5)
    assert len(r) == 3


def test_too_few_points():
    with pytest.raises(RouteError):
        Route([40, 40], [-105, -105])


def test_bad_gpx(tmp_path):
    p = tmp_path / "bad.gpx"
    p.write_text("not xml")
    with pytest.raises(RouteError):
        load_gpx(p)


def test_simplify_straight_line_collapses():
    pts = straight_north()
    r = Route(*zip(*pts))
    coords = simplified_coords(r, 50)
    assert len(coords) == 2
    assert coords[0] == pytest.approx(pts[0], abs=1e-7)
    assert coords[-1] == pytest.approx(pts[-1], abs=1e-7)


def test_point_at_midpoint():
    r = Route(*zip(*straight_north()))
    lat, lon = r.point_at(r.length_m / 2)
    assert lat == pytest.approx(40.25, abs=1e-3)
    assert lon == pytest.approx(-105.0, abs=1e-6)


def test_chunks_share_boundaries_and_respect_limits():
    coords = straight_north(step=0.01)  # 51 points, ~1.1 km apart
    chunks = chunk_coords(coords, max_points=10, max_length_m=1e9)
    assert all(len(c) <= 10 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a[-1] == b[0]
    # every input vertex is covered, in order
    flat = [chunks[0][0]] + [p for c in chunks for p in c[1:]]
    assert flat == coords


def test_chunks_respect_length():
    coords = straight_north(step=0.01)  # ~55 km
    chunks = chunk_coords(coords, max_points=1000, max_length_m=10_000)
    for c in chunks:
        lats = np.array([p[0] for p in c])
        assert (lats[-1] - lats[0]) * 111_000 <= 10_000 + 1
    assert len(chunks) >= 6
