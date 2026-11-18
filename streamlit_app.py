"""resupply web app (Streamlit).

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py
"""

from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # use the package straight from the repo on Streamlit Cloud

from resupply import __version__  # noqa: E402
from resupply.categories import CATEGORIES, FOOD, WATER  # noqa: E402
from resupply.extract import find_extract  # noqa: E402
from resupply.output import KM, MI, csv_text, gap_message, map_html, missing_message  # noqa: E402
from resupply.overpass import DiskCache, OverpassClient, OverpassError  # noqa: E402
from resupply.pipeline import Options, analyze, plan_chunks  # noqa: E402
from resupply.route import RouteError, parse_gpx  # noqa: E402

DEMOS = {
    "Burlington, VT → Boston, MA (229 mi, offline data: instant)": ROOT / "examples" / "burlington_boston.gpx",
    "Boulder → Lyons, CO (15 mi, live server: slower)": ROOT / "examples" / "boulder_lyons.gpx",
}
LONG_ROUTE_KM = 800

st.set_page_config(page_title="resupply", page_icon="💧", layout="wide")


@st.cache_resource
def overpass_cache() -> DiskCache:
    return DiskCache(Path(tempfile.gettempdir()) / "resupply-cache", ttl_s=7 * 86400)


@st.cache_resource
def offline_extract():
    try:
        return find_extract()
    except (OSError, ValueError):
        return None


@st.cache_resource(ttl=3600)
def unreachable_endpoints() -> set:
    """Shared across visitors for an hour, so one dead mirror isn't re-discovered by every run."""
    return set()


class StatusLog(logging.Handler):
    """Shows the tool's progress and retry messages inside an st.status box."""

    def __init__(self, box):
        super().__init__(logging.INFO)
        self.box = box

    def emit(self, record):
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            self.box.write(f"⚠️ {msg}")
        elif "features" in msg:
            self.box.write(msg)


# ---------------------------------------------------------------------------------------------
st.title("💧 resupply")
st.caption(
    "Upload a GPX route to find drinking water, groceries, convenience stores, gas stations, "
    "campsites and bike shops along it, and flag long stretches with no water or food. "
    "Data from OpenStreetMap."
)
_ex = offline_extract()
if _ex is not None:
    _names = ", ".join(
        "DC" if r.endswith("district-of-columbia") else r.rsplit("/", 1)[-1].replace("-", " ").title() for r in _ex.regions
    )
    with st.expander(f"⚡ Instant results in {len(_ex.regions)} northeastern US states (data from {_ex.data_date})"):
        st.write(f"Offline data covers: {_names}. Routes elsewhere use the live OpenStreetMap server, which can be slow.")

units_label = st.radio("Units", ["miles", "kilometres"], horizontal=True)
units = MI if units_label == "miles" else KM
mi = units is MI

with st.form("options"):
    upload = st.file_uploader("GPX route file", type=None, help="Export a .gpx from Ride with GPS, Strava, Komoot, Garmin…")
    demo = st.selectbox("No file handy? Try a demo route", ["—", *DEMOS], index=0)
    use_demo = demo != "—"

    labels = {c.label: c.key for c in CATEGORIES.values()}
    chosen = st.multiselect("What to look for", list(labels), default=list(labels))

    c1, c2 = st.columns(2)
    default_dist = c1.number_input(
        f"Search distance from route ({units.long})", min_value=0.01, value=0.6 if mi else 1.0, step=0.1, format="%.2f"
    )
    max_gap = c2.number_input(
        f"Flag gaps with no water or food longer than ({units.long})",
        min_value=0.0, value=30.0 if mi else 50.0, step=5.0,
        help="0 turns gap warnings off",
    )

    with st.expander("Different search distance per type"):
        st.caption("Leave blank to use the search distance above.")
        per_cat = {}
        for c in CATEGORIES.values():  # one column: two would stack out of order on phones
            per_cat[c.key] = st.number_input(
                f"{c.label} ({units.long})", min_value=0.01, value=None, step=0.1, format="%.2f", key=f"r_{c.key}"
            )

    with st.expander("More options"):
        fuel_as_food = st.checkbox(
            "Count gas stations as food & water", value=True,
            help="Most have a shop, but some rural pumps are card-only with nothing to buy.",
        )
        merge_dist = st.number_input(
            f"Merge features that are one stop, within ({units.short})",
            min_value=0.0, value=200.0 if mi else 60.0, step=10.0,
            help="E.g. a gas station and its convenience store mapped separately. 0 = never merge.",
        )
        refresh = st.checkbox("Use live map data only (skip offline data and cache)")

    submitted = st.form_submit_button("Find resupply points", type="primary", width="stretch")

# ---------------------------------------------------------------------------------------------
if submitted:
    if upload is None and not use_demo:
        st.error("Choose a GPX file, or pick a demo route.")
        st.stop()
    if not chosen:
        st.error("Pick at least one type of place to look for.")
        st.stop()

    try:
        if upload is not None:
            route = parse_gpx(upload.getvalue(), fallback_name=Path(upload.name).stem, source=upload.name)
            stem = Path(upload.name).stem
        else:
            route = parse_gpx(DEMOS[demo].read_bytes(), fallback_name="demo", source="demo route")
            stem = DEMOS[demo].stem
    except RouteError as e:
        st.error(f"Couldn't read that file: {e}")
        st.stop()

    categories = [CATEGORIES[labels[lbl]] for lbl in chosen]
    radii = {c.key: (per_cat[c.key] or default_dist) * units.long_m for c in categories}
    opts = Options(
        categories=categories,
        radii=radii,
        gap_limits={WATER: max_gap * units.long_m, FOOD: max_gap * units.long_m},
        fuel_as_food=fuel_as_food,
        merge_distance_m=merge_dist * units.short_m,
        refresh=refresh,
    )

    n_chunks = len(plan_chunks(route, opts))
    if route.length_m > LONG_ROUTE_KM * 1000:
        st.info(
            f"This is a long route ({units.dist(route.length_m, 0)} {units.long}, {n_chunks} map queries). "
            "It can take 10+ minutes when the public map server is busy — keep this page open."
        )

    client = OverpassClient(cache=overpass_cache())  # opts.refresh skips cached reads but saves fresh results
    client.unreachable = unreachable_endpoints()

    bar = st.progress(0.0, text=f"Querying OpenStreetMap: 0/{n_chunks}")
    with st.status("Working…", expanded=False) as box:
        handler = StatusLog(box)
        logger = logging.getLogger("resupply")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            result = analyze(
                route, opts, client, extract=None if refresh else offline_extract(),
                on_progress=lambda d, n: bar.progress(d / n, text=f"Querying OpenStreetMap: {d}/{n}"),
            )
        except OverpassError as e:
            box.update(label="Failed", state="error")
            st.error(f"The OpenStreetMap server couldn't be reached: {e}\n\nIt's often busy; try again in a few minutes.")
            st.stop()
        finally:
            logger.removeHandler(handler)
        box.update(label="Done", state="complete")
    bar.empty()

    st.session_state["out"] = {
        "result": result,
        "units": units,
        "stem": stem,
        "map": map_html(result.route, result.stops, result.gaps, units, result.missing),
        "csv": csv_text(result.stops, result.gaps, units, result.missing),
    }

# ---------------------------------------------------------------------------------------------
out = st.session_state.get("out")
if out:
    res, u = out["result"], out["units"]
    st.subheader(res.route.name)
    m1, m2, m3 = st.columns(3)
    m1.metric("Route", f"{u.dist(res.route.length_m)} {u.long}")
    m2.metric("Resupply points", res.n_places)
    m3.metric("Gaps flagged", len(res.gaps))
    if res.n_local_chunks == res.n_chunks:
        st.caption(f"⚡ All from offline data (OpenStreetMap data from {res.extract_date}).")
    elif res.n_local_chunks:
        st.caption(f"⚡ {res.n_local_chunks} of {res.n_chunks} stretches from offline data "
                   f"(OpenStreetMap data from {res.extract_date}); the rest queried live.")

    for a, b in res.missing:
        st.error(missing_message(a, b, u))
    for g in res.gaps:
        st.warning(f"{u.long} {u.dist(g.start_m)}: {gap_message(g, u)}")

    components.html(out["map"], height=520)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Cue sheet (CSV)", out["csv"], f"{out['stem']}_cues.csv", "text/csv",
                       width="stretch")
    d2.download_button("⬇️ Map (HTML)", out["map"], f"{out['stem']}_resupply.html", "text/html",
                       width="stretch")

    df = pd.read_csv(io.StringIO(out["csv"]), dtype=str).fillna("")
    keep = [f"dist_{u.long}", "category", "name", f"offset_{u.short}", "side",
            f"next_water_{u.long}", f"next_food_{u.long}", "opening_hours", "osm_url"]
    st.dataframe(
        df[keep],
        hide_index=True,
        width="stretch",
        column_config={
            f"dist_{u.long}": st.column_config.TextColumn(u.long, width="small"),
            f"offset_{u.short}": st.column_config.TextColumn(f"off ({u.short})", width="small"),
            "osm_url": st.column_config.LinkColumn("OSM", display_text="map"),
        },
    )

st.divider()
st.caption(
    f"resupply {__version__} · [source](https://github.com/tjmilich/resupply) · "
    "Map data © OpenStreetMap contributors (ODbL), via Geofabrik extracts and the public Overpass API. "
    "Your GPX file isn't stored; map results for the route are cached on the server for a few days. "
    "Check critical water sources yourself — map data can be missing or out of date."
)
