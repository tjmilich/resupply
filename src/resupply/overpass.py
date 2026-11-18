"""Overpass API query building, HTTP client with retries, and a disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Iterable

import requests

from . import __version__
from .categories import Category
from .route import LatLon

log = logging.getLogger(__name__)

DEFAULT_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
RETRY_STATUS = {429, 500, 502, 503, 504}
# Every query also counts highway ways this close to the chunk's polyline. A real route
# always runs along or near roads/paths, so a count of 0 means the server answered from a
# broken or empty database (observed on overpass-api.de: identical queries randomly
# returning zero elements with HTTP 200 and no error remark).
CANARY_RADIUS_M = 150


class OverpassError(RuntimeError):
    pass


class SuspectResponse(OverpassError):
    """The server kept answering with no data at all, not even roads near the route."""


def road_count(data: dict) -> int | None:
    """The canary's highway count from a response, or None if the response has no canary."""
    for el in data.get("elements", []):
        if el.get("type") == "count":
            try:
                return int(el["tags"]["ways"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def looks_valid(data: dict) -> bool:
    n = road_count(data)
    return n is None or n > 0


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "resupply"


def build_query(coords: list[LatLon], radius_m: float, categories: Iterable[Category], timeout_s: int = 180) -> str:
    """Build an Overpass QL query for all category filters around a polyline.

    ``around:R,lat1,lon1,lat2,lon2,...`` treats the coordinate list as a
    polyline, so the corridor follows the route rather than a bounding box.
    Coordinates are rounded to 6 dp so the text (and thus the cache key) is stable.
    """
    if len(coords) < 1:
        raise ValueError("need at least one coordinate")
    coord_str = ",".join(f"{lat:.6f},{lon:.6f}" for lat, lon in coords)
    around = f"(around:{radius_m:.0f},{coord_str})"
    lines = [f"[out:json][timeout:{timeout_s}];", "("]
    for cat in categories:
        for flt in cat.overpass_filters():
            lines.append(f"  nwr{flt}{around};")
    lines += [
        ");",
        "out center tags;",
        f'way["highway"](around:{CANARY_RADIUS_M},{coord_str});',
        "out count;",
    ]
    return "\n".join(lines)


class DiskCache:
    """One JSON file per query, keyed by SHA-256 of the query text."""

    def __init__(self, directory: str | Path, ttl_s: float | None = None):
        self.dir = Path(directory)
        self.ttl_s = ttl_s
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, query: str) -> dict | None:
        path = self._path(self.key(query))
        try:
            with path.open(encoding="utf-8") as f:
                entry = json.load(f)
        except (OSError, ValueError):
            return None
        if self.ttl_s is not None and time.time() - entry.get("fetched_at", 0) > self.ttl_s:
            return None
        return entry.get("response")

    def set(self, query: str, response: dict) -> None:
        path = self._path(self.key(query))
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "query": query, "response": response}, f)
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written entry


class OverpassClient:
    def __init__(
        self,
        endpoints: list[str] | None = None,
        cache: DiskCache | None = None,
        session: requests.Session | None = None,
        max_retries: int = 5,
        backoff_s: float = 5.0,
        min_interval_s: float = 1.0,
        http_timeout_s: float = 240.0,
        connect_timeout_s: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.endpoints = endpoints or list(DEFAULT_ENDPOINTS)
        self.cache = cache
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.min_interval_s = min_interval_s
        self.http_timeout_s = http_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self._sleep = sleep
        self._last_request = 0.0
        self._next = 0  # rotation position, kept across queries
        self.unreachable: set[str] = set()
        self.user_agent = f"resupply/{__version__} (GPX resupply finder; python-requests)"

    def is_cached(self, q: str) -> bool:
        if self.cache is None:
            return False
        cached = self.cache.get(q)
        return cached is not None and looks_valid(cached)

    def _pick_endpoint(self) -> str:
        """Next endpoint in rotation, skipping ones that timed out or refused connections this run.

        If every endpoint has been marked unreachable, all of them are tried again.
        """
        alive = [e for e in self.endpoints if e not in self.unreachable] or self.endpoints
        endpoint = alive[self._next % len(alive)]
        return endpoint

    def query(self, q: str, refresh: bool = False) -> tuple[dict, bool]:
        """Run a query. Returns (response, came_from_cache)."""
        if self.cache is not None and not refresh:
            cached = self.cache.get(q)
            if cached is not None and looks_valid(cached):
                return cached, True
        data = self._fetch(q)
        if self.cache is not None:
            self.cache.set(q, data)
        return data, False

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def _fetch(self, q: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            endpoint = self._pick_endpoint()
            self._throttle()
            try:
                resp = self.session.post(
                    endpoint,
                    data={"data": q},
                    timeout=(self.connect_timeout_s, self.http_timeout_s),
                    headers={"User-Agent": self.user_agent},
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                # A server that won't connect or never answers is skipped for the rest of the
                # run, so one dead mirror costs at most one timeout rather than one per query.
                last_err = e
                if len(self.endpoints) > 1 and endpoint not in self.unreachable:
                    self.unreachable.add(endpoint)
                    log.warning("%s is not responding; skipping it for the rest of this run", endpoint)
            except requests.RequestException as e:
                last_err = e
            else:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        last_err = OverpassError(f"non-JSON response from {endpoint}")
                    else:
                        # Overpass reports server-side timeouts/OOM as HTTP 200 with a remark;
                        # such results are partial and must not be cached.
                        remark = str(data.get("remark", ""))
                        if "error" in remark.lower():
                            last_err = OverpassError(f"{endpoint}: {remark.strip()}")
                        elif not looks_valid(data):
                            last_err = SuspectResponse(f"{endpoint} returned no data, not even roads near the route")
                        else:
                            return data
                elif resp.status_code in RETRY_STATUS:
                    last_err = OverpassError(f"HTTP {resp.status_code} from {endpoint}")
                else:
                    raise OverpassError(f"HTTP {resp.status_code} from {endpoint}: {resp.text[:500]}")
            self._next += 1  # rotate away from the endpoint that just failed
            if attempt < self.max_retries - 1:
                delay = self.backoff_s * (2**attempt)
                log.warning("Overpass request failed (%s); retrying in %.0fs", last_err, delay)
                self._sleep(delay)
        cls = SuspectResponse if isinstance(last_err, SuspectResponse) else OverpassError
        raise cls(f"Overpass query failed after {self.max_retries} attempts: {last_err}")


def fetch_elements(
    client: OverpassClient,
    chunks: list[list[LatLon]],
    radius_m: float,
    categories: list[Category],
    refresh: bool = False,
) -> tuple[list[dict], list[int]]:
    """Query every chunk.

    Returns OSM elements de-duplicated across chunks, and the 0-based indices of chunks
    for which no trustworthy data could be obtained (see ``SuspectResponse``). Those
    chunks are skipped rather than aborting the run; callers must report them.
    """
    seen: dict[tuple[str, int], dict] = {}
    missing: list[int] = []
    for i, chunk in enumerate(chunks, 1):
        q = build_query(chunk, radius_m, categories)
        if refresh or not client.is_cached(q):
            log.info("chunk %d/%d: querying Overpass...", i, len(chunks))
        try:
            data, cached = client.query(q, refresh=refresh)
        except SuspectResponse as e:
            log.warning("chunk %d/%d: no usable data (%s)", i, len(chunks), e)
            missing.append(i - 1)
            continue
        elements = [el for el in data.get("elements", []) if el.get("type") != "count"]
        log.info("chunk %d/%d: %d features%s", i, len(chunks), len(elements), " (cached)" if cached else "")
        for el in elements:
            seen.setdefault((el["type"], el["id"]), el)
    return list(seen.values()), missing
