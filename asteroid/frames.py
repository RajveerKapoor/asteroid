"""Time scales and coordinate-frame transforms.

Two jobs:

1. Convert between human calendar dates and Julian Date (JD), the continuous day
   count that orbital mechanics runs on. We use Meeus' algorithms (Astronomical
   Algorithms, ch. 7), valid across the Gregorian and Julian calendars.

2. Rotate vectors between the heliocentric **ecliptic** frame (where JPL orbital
   elements live) and the **equatorial** frame (where right ascension and
   declination live), and reduce Cartesian vectors to spherical angles.

Time-scale note: we treat UTC, TT and TDB as equal. At the arc-second precision
of a two-body propagator the <0.001-day differences are negligible; this is
documented as a limitation rather than corrected.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

import numpy as np

from .constants import OBLIQUITY_J2000_RAD


# --------------------------------------------------------------------------- #
# Calendar <-> Julian Date
# --------------------------------------------------------------------------- #
def calendar_to_jd(year: int, month: int, day: float) -> float:
    """Julian Date for a (proleptic) Gregorian calendar date.

    ``day`` may carry a fractional part to encode the time of day, e.g.
    ``day=1.5`` is noon on the 1st.
    """
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )


def jd_to_calendar(jd: float) -> tuple[int, int, int, int, int, float]:
    """Inverse of :func:`calendar_to_jd`.

    Returns ``(year, month, day, hour, minute, second)``.
    """
    jd = jd + 0.5
    z = math.floor(jd)
    f = jd - z
    if z < 2299161:
        a = z
    else:
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day_frac = b - d - math.floor(30.6001 * e) + f
    day = int(day_frac)
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715

    frac = day_frac - day
    seconds_total = frac * 86400.0
    hour = int(seconds_total // 3600)
    minute = int((seconds_total - hour * 3600) // 60)
    second = seconds_total - hour * 3600 - minute * 60
    return year, month, day, hour, minute, second


def datetime_to_jd(dt: datetime) -> float:
    """Julian Date for a :class:`datetime` (naive datetimes are treated as UTC)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    day = dt.day + (
        dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60.0) / 60.0
    ) / 24.0
    return calendar_to_jd(dt.year, dt.month, day)


def jd_to_datetime(jd: float) -> datetime:
    """:class:`datetime` (UTC) for a Julian Date, rounded to the nearest second.

    Rounding to whole seconds matches this tool's precision regime and, by
    building the result with :class:`timedelta`, lets sub-second arithmetic
    error carry cleanly across minute/hour/day boundaries instead of, say,
    turning ``21:46:00`` into ``21:45:59.9999``.
    """
    year, month, day, hour, minute, second = jd_to_calendar(jd)
    base = datetime(year, month, day, tzinfo=timezone.utc)
    seconds_of_day = hour * 3600 + minute * 60 + second
    return base + timedelta(seconds=round(seconds_of_day))


def now_jd() -> float:
    """Julian Date of the current instant (UTC)."""
    return datetime_to_jd(datetime.now(timezone.utc))


_ISO_RE = re.compile(
    r"^(\d{4})-(\d{1,2})-(\d{1,2})"
    r"(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}(?:\.\d+)?))?)?$"
)


def parse_date(text: str) -> float:
    """Parse a user date string into a Julian Date.

    Accepts:
      * ``now`` / ``today``
      * ``YYYY-MM-DD`` and ``YYYY-MM-DD[ T]HH:MM[:SS]``
      * ``JD2451545.0`` or a bare Julian Date number (>= 100000)
    """
    s = text.strip()
    low = s.lower()
    if low in ("now", "today"):
        return now_jd()
    if low.startswith("jd"):
        return float(s[2:])

    m = _ISO_RE.match(s)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hour = int(m.group(4)) if m.group(4) else 0
        minute = int(m.group(5)) if m.group(5) else 0
        second = float(m.group(6)) if m.group(6) else 0.0
        day_frac = day + (hour + (minute + second / 60.0) / 60.0) / 24.0
        return calendar_to_jd(year, month, day_frac)

    # Bare number: small values are ambiguous, so require a Julian-Date scale.
    try:
        val = float(s)
    except ValueError as exc:
        raise ValueError(f"Could not parse date: {text!r}") from exc
    if val < 100000:
        raise ValueError(
            f"Ambiguous date {text!r}; use YYYY-MM-DD or an explicit JD (>=100000)."
        )
    return val


def format_jd(jd: float, with_time: bool = True) -> str:
    """Render a Julian Date as an ISO-ish calendar string (UTC)."""
    year, month, day, hour, minute, second = jd_to_calendar(jd)
    if not with_time:
        return f"{year:04d}-{month:02d}-{day:02d}"
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{int(second):02d} UTC"


# --------------------------------------------------------------------------- #
# Ecliptic <-> Equatorial (rotation about the shared x-axis by the obliquity)
# --------------------------------------------------------------------------- #
def ecliptic_to_equatorial(vec: np.ndarray, obliquity: float = OBLIQUITY_J2000_RAD) -> np.ndarray:
    """Rotate a 3-vector from ecliptic to equatorial coordinates."""
    c, s = math.cos(obliquity), math.sin(obliquity)
    x, y, z = vec[0], vec[1], vec[2]
    return np.array([x, c * y - s * z, s * y + c * z])


def equatorial_to_ecliptic(vec: np.ndarray, obliquity: float = OBLIQUITY_J2000_RAD) -> np.ndarray:
    """Rotate a 3-vector from equatorial to ecliptic coordinates."""
    c, s = math.cos(obliquity), math.sin(obliquity)
    x, y, z = vec[0], vec[1], vec[2]
    return np.array([x, c * y + s * z, -s * y + c * z])


def cartesian_to_spherical(vec: np.ndarray) -> tuple[float, float, float]:
    """Return ``(radius, longitude_deg, latitude_deg)`` for a Cartesian vector.

    Longitude is measured in the x-y plane in ``[0, 360)``; latitude in
    ``[-90, 90]``. For an ecliptic vector this gives ecliptic lon/lat; for an
    equatorial vector it gives (RA, Dec).
    """
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    r = math.sqrt(x * x + y * y + z * z)
    lon = math.degrees(math.atan2(y, x)) % 360.0
    lat = math.degrees(math.asin(z / r)) if r > 0 else 0.0
    return r, lon, lat


def equatorial_to_radec(vec: np.ndarray) -> tuple[float, float]:
    """Return ``(ra_deg, dec_deg)`` for an equatorial Cartesian vector."""
    _, ra, dec = cartesian_to_spherical(vec)
    return ra, dec


def gmst_deg(jd: float) -> float:
    """Greenwich Mean Sidereal Time (degrees) for a Julian Date.

    IAU-1982 polynomial. We treat the input JD (UTC) as UT1; the <1 s difference
    is far below this tool's precision. Used to orient ground observatories in
    inertial space for topocentric corrections.
    """
    t = (jd - 2451545.0) / 36525.0
    gmst = (280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * t * t
            - t * t * t / 38_710_000.0)
    return gmst % 360.0


def format_ra(ra_deg: float) -> str:
    """Format a right ascension (degrees) as ``HHh MMm SSs``."""
    hours = (ra_deg % 360.0) / 15.0
    h = int(hours)
    m = int((hours - h) * 60)
    s = (hours - h - m / 60.0) * 3600
    return f"{h:02d}h {m:02d}m {s:04.1f}s"


def format_dec(dec_deg: float) -> str:
    """Format a declination (degrees) as ``+DD° MM' SS\"``."""
    sign = "+" if dec_deg >= 0 else "-"
    a = abs(dec_deg)
    d = int(a)
    m = int((a - d) * 60)
    s = (a - d - m / 60.0) * 3600
    return f"{sign}{d:02d}° {m:02d}' {s:04.1f}\""
