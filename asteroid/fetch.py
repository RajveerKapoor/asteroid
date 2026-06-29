"""Live data from NASA/JPL.

Two services are used:

* **SBDB** (Small-Body Database) — :func:`fetch_sbdb` returns a fully populated
  :class:`~asteroid.database.Body` (osculating elements + physical data) for any
  asteroid or comet by name or designation.
* **Horizons** — :func:`fetch_horizons_vector` returns NASA's *full-perturbation*
  heliocentric state vector for a body at an instant, used by ``--validate`` to
  measure how far our two-body propagation drifts from the real ephemeris.

All network calls have timeouts and raise :class:`FetchError` on any failure so
the CLI can fall back to the offline cache with a clear message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import requests

from . import frames
from .constants import AU_KM
from .database import Body, db_home, normalize_key

SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SENTRY_URL = "https://ssd-api.jpl.nasa.gov/sentry.api"
MPC_OBS_URL = "https://data.minorplanetcenter.net/api/get-obs"
OBSCODES_URL = "https://www.minorplanetcenter.net/iau/lists/ObsCodes.html"
USER_AGENT = "asteroid-trajectory/0.1 (educational orbital mechanics CLI)"
TIMEOUT = 25

# Parallax constants (longitude_east_deg, rho*cos(phi'), rho*sin(phi')) for a few
# prolific stations, used as an offline fallback if the MPC list can't be fetched.
# Verified from the MPC observatory-code list; the live fetch supplies all others.
_FALLBACK_OBSCODES = {
    "500": (0.0, 0.0, 0.0),                    # geocentre
    "691": (248.39966, 0.849466, 0.526479),    # Spacewatch, Kitt Peak
    "703": (249.26736, 0.845311, 0.533211),    # Catalina Sky Survey
    "G96": (249.21128, 0.845107, 0.533611),    # Mt. Lemmon Survey
    "F51": (203.74409, 0.936241, 0.351543),    # Pan-STARRS 1, Haleakala
}


class FetchError(RuntimeError):
    """Raised when a remote lookup fails or returns no usable object."""


def _get(url: str, params: dict) -> dict:
    try:
        resp = requests.get(url, params=params,
                            headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise FetchError(f"network error contacting {url}: {exc}") from exc
    if resp.status_code != 200:
        raise FetchError(f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise FetchError(f"invalid JSON from {url}: {exc}") from exc


def _f(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_sbdb(query: str) -> Body:
    """Fetch one object from the JPL Small-Body Database as a :class:`Body`."""
    data = _get(SBDB_URL, {"sstr": query, "full-prec": "1", "phys-par": "1"})

    if "object" not in data or "orbit" not in data:
        # SBDB returns a candidate list when the search is ambiguous.
        if "list" in data:
            names = ", ".join(item.get("name", "?") for item in data["list"][:8])
            raise FetchError(
                f"{query!r} is ambiguous; candidates include: {names}"
            )
        msg = data.get("message", "object not found")
        raise FetchError(f"JPL SBDB: {msg} (query={query!r})")

    obj = data["object"]
    orbit = data["orbit"]
    elements = {el["name"]: el.get("value") for el in orbit.get("elements", [])}

    phys = {p["name"]: p.get("value") for p in data.get("phys_par", [])}

    orbit_class = ""
    oc = obj.get("orbit_class")
    if isinstance(oc, dict):
        orbit_class = oc.get("name", "") or oc.get("code", "")

    a = _f(elements.get("a"))
    e = _f(elements.get("e"))
    if a is None or e is None:
        raise FetchError(f"JPL SBDB returned incomplete elements for {query!r}")

    name = obj.get("shortname") or obj.get("fullname") or query

    return Body(
        key=normalize_key(name),
        name=name,
        a=a,
        e=e,
        inc_deg=_f(elements.get("i")) or 0.0,
        node_deg=_f(elements.get("om")) or 0.0,
        argp_deg=_f(elements.get("w")) or 0.0,
        M0_deg=_f(elements.get("ma")) or 0.0,
        epoch=_f(orbit.get("epoch")) or 0.0,
        n_deg=_f(elements.get("n")),
        per_days=_f(elements.get("per")),
        q=_f(elements.get("q")),
        ad=_f(elements.get("ad")),
        H=_f(phys.get("H")),
        albedo=_f(phys.get("albedo")),
        diameter_km=_f(phys.get("diameter")),
        neo=bool(obj.get("neo")),
        pha=bool(obj.get("pha")),
        orbit_class=orbit_class,
        moid=_f(orbit.get("moid")),
        fullname=obj.get("fullname", ""),
        designation=obj.get("des", ""),
        spkid=str(obj.get("spkid", "")),
        source="JPL SBDB",
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    )


# --------------------------------------------------------------------------- #
# Sentry: NASA/JPL's impact-risk list — the "open problems" of NEO science.
# Every object here has a non-zero computed probability of hitting Earth; the
# list is what planetary-defense researchers actually watch. Each row carries a
# designation that resolves straight through fetch_sbdb, so the rest of this
# tool can compute any of them.
# --------------------------------------------------------------------------- #
@dataclass
class SentryRisk:
    """One object on the JPL Sentry impact-risk list."""

    designation: str            # e.g. "2000 SG344" — resolves via fetch_sbdb
    fullname: str
    impact_prob: float          # cumulative impact probability (ip)
    palermo_cum: float          # cumulative Palermo Technical Scale (the ranking)
    palermo_max: float          # single-encounter Palermo maximum
    torino_max: int             # maximum Torino Scale (0 unless newsworthy)
    diameter_km: Optional[float]
    h: Optional[float]
    v_inf: Optional[float]      # velocity relative to Earth at encounter (km/s)
    n_impacts: int              # number of distinct potential impacts found
    year_range: str             # span of potential impact years, e.g. "2056-2113"
    last_obs: str

    @property
    def name(self) -> str:
        return self.designation


@dataclass
class SentryObject:
    """Sentry's detailed risk summary for a single object."""

    designation: str
    fullname: str
    impact_prob: float
    palermo_cum: float
    torino_max: int
    diameter_km: Optional[float]
    v_impact_km_s: Optional[float]
    energy_mt: Optional[float]   # kinetic energy at impact (megatons TNT)
    n_impacts: int
    year_range: str
    n_obs: Optional[int]
    arc_days: Optional[float]


def _parse_sentry_row(row: dict) -> SentryRisk:
    return SentryRisk(
        designation=str(row.get("des", "")).strip(),
        fullname=str(row.get("fullname", "")).strip(),
        impact_prob=_f(row.get("ip")) or 0.0,
        palermo_cum=_f(row.get("ps_cum")) or -99.0,
        palermo_max=_f(row.get("ps_max")) or -99.0,
        torino_max=int(_f(row.get("ts_max")) or 0),
        diameter_km=_f(row.get("diameter")),
        h=_f(row.get("h")),
        v_inf=_f(row.get("v_inf")),
        n_impacts=int(_f(row.get("n_imp")) or 0),
        year_range=str(row.get("range", "")).strip(),
        last_obs=str(row.get("last_obs", "")).strip(),
    )


def fetch_sentry_list(limit: Optional[int] = None) -> List[SentryRisk]:
    """The JPL Sentry impact-risk list, ranked most-hazardous first.

    Sorted by cumulative Palermo scale (the standard hazard ranking). Pass
    ``limit`` to keep only the top entries. Raises :class:`FetchError` on failure.
    """
    data = _get(SENTRY_URL, {})
    rows = data.get("data")
    if not rows:
        raise FetchError("JPL Sentry returned no risk objects")
    risks = [_parse_sentry_row(r) for r in rows]
    risks.sort(key=lambda r: r.palermo_cum, reverse=True)
    return risks[:limit] if limit else risks


def _year_range_from_impacts(impacts: list) -> str:
    years = []
    for entry in impacts:
        date = str(entry.get("date", ""))
        if len(date) >= 4 and date[:4].isdigit():
            years.append(int(date[:4]))
    if not years:
        return ""
    return f"{min(years)}" if min(years) == max(years) else f"{min(years)}-{max(years)}"


def fetch_sentry_object(designation: str) -> Optional[SentryObject]:
    """Sentry's risk summary for one object, or ``None`` if it carries no risk.

    Objects not on the Sentry list (the overwhelming majority) return ``None``
    rather than raising — "no known impact risk" is a normal, useful answer.
    Sentry replies HTTP 400 "invalid designation" for objects it doesn't track,
    which is treated as the same no-risk answer.
    """
    try:
        data = _get(SENTRY_URL, {"des": designation})
    except FetchError as exc:
        if "invalid designation" in str(exc).lower() or "HTTP 400" in str(exc):
            return None
        raise
    summary = data.get("summary")
    if not summary:
        return None
    impacts = data.get("data", []) or []
    return SentryObject(
        designation=str(summary.get("des", designation)).strip(),
        fullname=str(summary.get("fullname", "")).strip(),
        impact_prob=_f(summary.get("ip")) or 0.0,
        palermo_cum=_f(summary.get("ps_cum")) or -99.0,
        torino_max=int(_f(summary.get("ts_max")) or 0),
        diameter_km=_f(summary.get("diameter")),
        v_impact_km_s=_f(summary.get("v_imp")),
        energy_mt=_f(summary.get("energy")),
        n_impacts=int(_f(summary.get("n_imp")) or len(impacts)),
        year_range=_year_range_from_impacts(impacts),
        n_obs=int(_f(summary.get("nobs")) or 0) or None,
        arc_days=_f(summary.get("darc")),
    )


def fetch_horizons_vector(query: str, jd: float) -> np.ndarray:
    """NASA Horizons heliocentric ecliptic position (AU) of ``query`` at ``jd``.

    Returns a 3-vector in the J2000 ecliptic frame (Horizons ``OUT_UNITS=AU-D``,
    ``REF_PLANE=ECLIPTIC``, center = Sun) — directly comparable to our own
    two-body position. The instant is given as a Julian Date interpreted in
    Horizons' default TDB scale, matching this tool's TDB-equivalent timekeeping.
    Raises :class:`FetchError` on any failure.
    """
    params = {
        "format": "json",
        "COMMAND": f"'{_horizons_command(query)}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "'500@10'",            # Sun body centre
        "REF_PLANE": "ECLIPTIC",
        "REF_SYSTEM": "J2000",
        "VEC_TABLE": "1",                # position only
        "OUT_UNITS": "AU-D",
        "TLIST_TYPE": "JD",
        "TLIST": f"{jd:.8f}",
    }
    data = _get(HORIZONS_URL, params)
    result = data.get("result", "")
    return _parse_horizons_vectors(result)


def _horizons_command(query: str) -> str:
    """Map a name to a Horizons small-body designator.

    Horizons resolves most asteroid names directly; appending ``;`` forces a
    small-body (rather than major-body) lookup, avoiding planet-name clashes.
    """
    q = query.strip()
    if q.endswith(";"):
        return q
    return f"{q};"


# --------------------------------------------------------------------------- #
# Minor Planet Center: observatory codes and observations
# --------------------------------------------------------------------------- #
def parse_obscodes(text: str) -> dict:
    """Parse the MPC observatory-code list into ``{code: (lon, cos, sin)}``.

    Fixed-width columns: code [0:3], longitude east [4:13], rho*cos(phi') [13:21],
    rho*sin(phi') [21:30], name [30:]. Lines without parallax constants
    (space telescopes, roving observers) are skipped.
    """
    if "<pre>" in text:
        text = text.split("<pre>", 1)[1].split("</pre>", 1)[0]
    codes = {}
    for line in text.splitlines():
        if len(line) < 30 or line.startswith("Code"):
            continue
        code = line[0:3].strip()
        if not code:
            continue
        try:
            lon = float(line[4:13])
            cos = float(line[13:21])
            sin = float(line[21:30])
        except ValueError:
            continue  # space-based / no fixed parallax constants
        codes[code] = (lon, cos, sin)
    return codes


def fetch_obscodes() -> dict:
    """Download and parse the MPC observatory-code list."""
    try:
        resp = requests.get(OBSCODES_URL, headers={"User-Agent": USER_AGENT},
                            timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise FetchError(f"network error fetching observatory codes: {exc}") from exc
    if resp.status_code != 200:
        raise FetchError(f"observatory codes returned HTTP {resp.status_code}")
    codes = parse_obscodes(resp.text)
    if not codes:
        raise FetchError("could not parse any observatory codes")
    return codes


def load_obscodes(force_refresh: bool = False) -> dict:
    """Observatory codes, cached under ASTEROID_HOME; falls back if offline."""
    cache = db_home() / "obscodes.json"
    if cache.exists() and not force_refresh:
        try:
            return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
        except (ValueError, OSError):
            pass
    try:
        codes = fetch_obscodes()
        cache.write_text(json.dumps(codes))
        return codes
    except FetchError:
        return dict(_FALLBACK_OBSCODES)


def _parse_obs80_line(line):
    """Parse one MPC 80-column observation line into ``(jd, ra_deg, dec_deg, stn)``.

    Returns ``None`` for radar/satellite/roving continuation lines or anything
    that doesn't carry a plain optical RA/Dec. Column layout (1-indexed): date
    16-32, RA 33-44, Dec 45-56, observatory code 78-80.
    """
    if len(line) < 80:
        return None
    note2 = line[14]
    if note2 in ("s", "r", "R", "v"):    # second lines / radar — no optical RA/Dec
        return None
    try:
        y, mo, day = line[15:32].split()
        jd = frames.calendar_to_jd(int(y), int(mo), float(day))
        rh, rm, rs = line[32:44].split()
        ra_deg = (int(rh) + int(rm) / 60.0 + float(rs) / 3600.0) * 15.0
        dec_field = line[44:56].strip()
        sign = -1.0 if dec_field[0] == "-" else 1.0
        dd, dm, ds = dec_field.lstrip("+-").split()
        dec_deg = sign * (int(dd) + int(dm) / 60.0 + float(ds) / 3600.0)
    except (ValueError, IndexError):
        return None
    return jd, ra_deg, dec_deg, line[77:80].strip()


def fetch_mpc_observations(name: str, max_records: Optional[int] = None) -> List:
    """Fetch an object's astrometry from the MPC as a list of ``iod.Observation``.

    Uses the compact 80-column format (≈20x smaller than ADES). Optical
    observations only; each gets a topocentric observer position from its station
    code. Raises :class:`FetchError` if nothing usable is found.
    """
    from .iod import Observation, observer_position_topocentric

    body = {"desigs": [name], "output_format": ["OBS80"]}
    try:
        resp = requests.get(MPC_OBS_URL, json=body,
                            headers={"User-Agent": USER_AGENT}, timeout=60)
    except requests.RequestException as exc:
        raise FetchError(f"network error contacting MPC: {exc}") from exc
    if resp.status_code != 200:
        raise FetchError(f"MPC get-obs returned HTTP {resp.status_code}: "
                         f"{resp.text[:200]}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise FetchError(f"invalid JSON from MPC: {exc}") from exc

    block = data[0].get("OBS80") if isinstance(data, list) and data else None
    if not block:
        raise FetchError(f"MPC has no observations for {name!r}")
    lines = block.splitlines() if isinstance(block, str) else list(block)

    obscodes = load_obscodes()
    observations = []
    for line in lines:
        parsed = _parse_obs80_line(line)
        if parsed is None:
            continue
        jd, ra_deg, dec_deg, stn = parsed
        observer = observer_position_topocentric(stn, jd, obscodes)
        observations.append(Observation(jd=jd, ra_deg=ra_deg, dec_deg=dec_deg,
                                        observer=observer))
    if not observations:
        raise FetchError(f"no usable optical observations for {name!r}")
    if max_records and len(observations) > max_records:
        step = len(observations) / max_records
        observations = [observations[int(i * step)] for i in range(max_records)]
    return observations


def _parse_horizons_vectors(result: str) -> np.ndarray:
    """Extract the X/Y/Z position (AU) from a Horizons VECTORS text block."""
    if "$$SOE" not in result or "$$EOE" not in result:
        raise FetchError(f"Horizons returned no ephemeris:\n{result[:300]}")
    block = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    x = y = z = None
    for token in ("X", "Y", "Z"):
        marker = f"{token} ="
        idx = block.find(marker)
        if idx == -1:
            raise FetchError("Horizons vector missing component " + token)
        tail = block[idx + len(marker):].strip()
        value = tail.split()[0].replace("D", "E")
        val = float(value)
        if token == "X":
            x = val
        elif token == "Y":
            y = val
        else:
            z = val
    return np.array([x, y, z])
