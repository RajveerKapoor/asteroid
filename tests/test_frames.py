"""Tests for time scales and coordinate-frame transforms."""

import math

import numpy as np
import pytest

from asteroid import frames
from asteroid.constants import OBLIQUITY_J2000_RAD, J2000_JD


def test_j2000_julian_date():
    """2000-01-01 12:00 is exactly JD 2451545.0 by definition."""
    assert frames.calendar_to_jd(2000, 1, 1.5) == pytest.approx(J2000_JD, abs=1e-9)


def test_calendar_round_trip():
    for jd in [2451545.0, 2461200.5, 2400000.5, 2469807.5]:
        y, mo, d, h, mi, s = frames.jd_to_calendar(jd)
        back = frames.calendar_to_jd(y, mo, d + (h + (mi + s / 60) / 60) / 24)
        assert back == pytest.approx(jd, abs=1e-7)


def test_datetime_round_trip():
    jd = frames.parse_date("2029-04-13 21:46:00")
    dt = frames.jd_to_datetime(jd)
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2029, 4, 13, 21, 46)


def test_parse_date_forms():
    midnight = frames.parse_date("2026-06-29")
    noon = frames.parse_date("2026-06-29 12:00")
    assert noon - midnight == pytest.approx(0.5, abs=1e-9)
    assert frames.parse_date("JD2451545.0") == pytest.approx(2451545.0)
    assert frames.parse_date("2451545.0") == pytest.approx(2451545.0)


def test_parse_date_rejects_ambiguous_number():
    with pytest.raises(ValueError):
        frames.parse_date("42")


def test_ecliptic_equatorial_round_trip():
    rng = np.array([0.7, -0.3, 0.55])
    out = frames.equatorial_to_ecliptic(frames.ecliptic_to_equatorial(rng))
    assert np.allclose(out, rng, atol=1e-12)


def test_vernal_equinox_axis_invariant():
    """The shared x-axis (vernal equinox) is unchanged by the obliquity rotation."""
    x = np.array([1.0, 0.0, 0.0])
    assert np.allclose(frames.ecliptic_to_equatorial(x), x, atol=1e-12)


def test_ecliptic_pole_maps_to_obliquity():
    """The ecliptic y-axis tilts to declination = obliquity in the equatorial frame."""
    y = np.array([0.0, 1.0, 0.0])
    _, dec = frames.equatorial_to_radec(frames.ecliptic_to_equatorial(y))
    assert math.radians(dec) == pytest.approx(OBLIQUITY_J2000_RAD, abs=1e-9)


def test_radec_formatting():
    assert frames.format_ra(0.0).startswith("00h")
    assert frames.format_ra(180.0).startswith("12h")
    assert frames.format_dec(-23.5).startswith("-23")
