from __future__ import annotations

import json

import numpy as np
import pytest


def gpx_text(points, name="Test route") -> str:
    pts = "\n".join(f'<trkpt lat="{lat:.7f}" lon="{lon:.7f}"></trkpt>' for lat, lon in points)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">\n'
        f"<trk><name>{name}</name><trkseg>\n{pts}\n</trkseg></trk></gpx>\n"
    )


def straight_north(lat0=40.0, lat1=40.5, lon=-105.0, step=0.001):
    lats = np.arange(lat0, lat1 + step / 2, step)
    return [(float(la), lon) for la in lats]


@pytest.fixture
def write_gpx(tmp_path):
    def _write(points, name="Test route", filename="route.gpx"):
        p = tmp_path / filename
        p.write_text(gpx_text(points, name), encoding="utf-8")
        return p

    return _write


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Returns queued responses; records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, data=None, timeout=None, headers=None):
        self.calls.append((url, data))
        return self.responses.pop(0)


def node(osm_id, lat, lon, **tags):
    return {"type": "node", "id": osm_id, "lat": lat, "lon": lon, "tags": tags}
