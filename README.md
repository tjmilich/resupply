# resupply

Find drinking water, supermarkets, convenience stores, gas stations, campsites and bike
shops along and near a GPX route using OpenStreetMap data from the Overpass API. resupply produces:

1. **An interactive map** (folium/Leaflet HTML). It shows the route, distance markers, highlighted gaps, and one
   toggleable layer per category. Each popup gives distance along the route, offset, hours, next water/food and an OSM link.
2. **A CSV cue sheet** sorted by distance along the route. Each row has the offset, the side of the route (L/R),
   what the stop provides, the distance to the next water and food, and the opening hours. Gap warnings appear as rows at the point where each gap starts.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

## Usage

```bash
resupply ride.gpx                        # writes ride_resupply.html + ride_cues.csv next to the GPX
resupply ride.gpx -o out/ --units mi --max-gap 40
resupply ride.gpx -r 500 -c water,supermarket,convenience --max-water-gap 25 --max-food-gap 60
resupply ride.gpx --refresh              # re-query Overpass, ignoring cached responses
```

Key options (`resupply -h` lists them all):

| option | default | meaning |
|---|---|---|
| `-r/--radius` | 1000 m | max offset from the route |
| `-c/--categories` | all | `water,supermarket,fuel,convenience,campsite,bike_shop` |
| `--max-gap` | 50 km / 30 mi | flag stretches with no water **or** no food longer than this |
| `--max-water-gap`, `--max-food-gap` | = `--max-gap` | separate thresholds per resource (0 disables) |
| `--merge-distance` | 60 m | merge features that are one real-world stop (0 = off) |
| `--no-fuel-food` | off | don't count gas stations as food/water unless they're tagged with a shop |
| `--simplify` | 50 m | Douglas–Peucker tolerance used for the query polyline |
| `--chunk-points`, `--chunk-km` | 150, 50 | per-query limits |
| `--cache-dir`, `--cache-ttl-days` | `~/.cache/resupply`, 30 | disk cache |

## How it works

```
GPX ──► Route (geodesic cumulative distance + local AEQD projection)
          │
          ├─► simplify (metres) ─► densify long segments ─► chunk (shared boundary vertex)
          │                                                     │
          │                       Overpass: nwr[filter](around:R+tol, lat,lon,lat,lon,…)
          │                       disk cache keyed by SHA-256(query), retry/backoff, endpoint fallback
          │                                                     │
          ├─◄──────── classify + dedupe elements (node/way centre) ◄┘
          │
          ├─► snap to FULL-resolution route: STRtree dwithin → nearest segment per "pass"
          │     along = geodesic cum distance + t·segment, offset = geodesic, side = cross product
          ├─► next water / next food per stop, gaps per resource (start & finish are boundaries)
          └─► CSV + folium map
```

Design notes:

- **Accuracy:** the projection is only used to search for the nearest segment. Distance along the route and offset are
  computed on the WGS84 ellipsoid, so they stay accurate on long routes.
- **Simplification only affects the query.** The `around` radius is widened by the simplification tolerance so nothing
  is missed. Results are then filtered against the true route with the requested radius.
- **Multiple passes:** on out-and-back, loop and figure-eight routes, a POI next to two different parts of the route
  gets one cue row per pass. The map shows one marker with all the distances.
- **Merging duplicates:** OpenStreetMap often maps one stop as several features, such as a fuel station, its shop
  and individual pump nodes. Features within `--merge-distance` (60 m) on the same pass of the route become one cue
  row, for example "Love's: Gas station + Convenience store". Separately named businesses never merge. Water,
  campsites and bike shops merge only with their own kind. A feature must also be within 2× the merge distance of the
  cluster's first feature, so a dense strip can't chain into one stop. Use `--merge-distance 0` to turn merging off.
- **Category definitions** (`categories.py`) are structured tag conditions. They generate the Overpass filters and also
  classify the returned elements, so the two can't disagree. Features tagged `access=private|no` are skipped.
- **What counts for gap analysis:** water taps and springs count as water. Supermarkets and convenience stores count as
  food and water. Gas stations count as food and water by default (`--no-fuel-food` turns that off). Campsites count as
  water only when tagged `drinking_water=yes`. Bike shops don't count for either.
- **Cache:** responses are written atomically. Overpass runtime-error responses (HTTP 200 with an error `remark`) are
  retried and never cached.
- **Bad-data detection:** public Overpass servers sometimes return an empty result that looks valid (HTTP 200, no
  error). To catch this, each query also counts the roads within 150 m of the chunk. A count of zero means the answer
  is bogus, so it's retried and never cached. If a stretch still has no usable data after the retries, it is marked
  **NO DATA** in the CSV and on the map. Gaps that overlap it are labelled "may be wrong", and the tool exits with
  code **3**.
- **Failover:** a server that refuses the connection or never answers is skipped for the rest of the run. A
  server that is only overloaded (429/5xx) is retried with backoff.
- **Map tiles:** the default background is CyclOSM. `tile.openstreetmap.org` blocks maps opened from a local file,
  because the browser sends no Referer. Standard OSM is still in the layer menu and works if the map is served over HTTP.

Exit codes: `0` success, `1` error (bad GPX, Overpass unreachable), `2` bad arguments, `3` finished but some
stretches had no usable map data.

## Layout

```
src/resupply/
  cli.py         argument parsing and orchestration
  categories.py  category definitions → Overpass filters / classifier / provides
  route.py       GPX loading, projection, geodesic distances, simplify, densify, chunk
  overpass.py    query builder, HTTP client (retry, throttle, fallback), DiskCache
  analysis.py    POIs, snapping/multi-pass, next water/food, gaps
  output.py      CSV cue sheet and folium map
tests/           offline tests (fake Overpass session, synthetic routes)
examples/        demo GPX (Boulder → Lyons, CO)
```

```bash
.venv/bin/pytest
```

OSM data © OpenStreetMap contributors (ODbL). Please be considerate of the public Overpass servers: the cache exists so
you don't re-query unchanged routes.
