"""Tests for observable quantities and the close-approach scanner."""

import math

import pytest

from asteroid import frames, observe
from asteroid.propagate import OrbitalElements


def apophis() -> OrbitalElements:
    """Apophis osculating elements (JPL SBDB, epoch JD 2461200.5)."""
    return OrbitalElements.from_degrees(
        a=0.9223592206975018, e=0.1911492279663492,
        inc_deg=3.340996879880978, node_deg=203.8936514240762,
        argp_deg=126.6795706895841, M0_deg=175.3304026592739,
        epoch=2461200.5, n_deg=1.112638115271892, name="Apophis",
    )


def ceres() -> OrbitalElements:
    return OrbitalElements.from_degrees(
        a=2.7656, e=0.0797, inc_deg=10.59, node_deg=80.27,
        argp_deg=73.6, M0_deg=130.0, epoch=2461000.5, name="Ceres",
    )


def test_observation_ranges_make_sense():
    obs = observe.observe(ceres(), frames.parse_date("2026-06-29"), H=3.3)
    # Ceres heliocentric distance is within its orbit.
    assert 2.5 < obs.r_helio < 3.0
    # Earth distance bounded by (r - 1) and (r + 1).
    assert obs.r_helio - 1.05 < obs.delta < obs.r_helio + 1.05
    assert 0.0 <= obs.ra < 360.0
    assert -90.0 <= obs.dec <= 90.0
    assert 0.0 <= obs.elongation <= 180.0
    assert 0.0 <= obs.phase_angle <= 180.0
    assert obs.magnitude is not None and 5.0 < obs.magnitude < 12.0


def test_magnitude_at_zero_phase():
    # tan(0) = 0 -> phase term vanishes -> V = H + 5 log10(r*delta).
    v = observe.phase_integral_magnitude(H=10.0, r=1.0, delta=1.0, alpha_rad=0.0)
    assert v == pytest.approx(10.0, abs=1e-9)
    v2 = observe.phase_integral_magnitude(H=10.0, r=2.0, delta=3.0, alpha_rad=0.0)
    assert v2 == pytest.approx(10.0 + 5.0 * math.log10(6.0), abs=1e-9)


def test_closer_is_brighter():
    near = observe.phase_integral_magnitude(15.0, 1.0, 0.1, 0.2)
    far = observe.phase_integral_magnitude(15.0, 1.0, 1.0, 0.2)
    assert near < far  # smaller magnitude == brighter


def test_geocentric_distance_matches_observation():
    jd = frames.parse_date("2029-04-13")
    d1 = observe.geocentric_distance(apophis(), jd)
    d2 = observe.observe(apophis(), jd).delta
    assert d1 == pytest.approx(d2, rel=1e-12)


def test_apophis_2029_close_approach():
    """The scanner must surface the 2029 flyby as the closest approach."""
    start = frames.parse_date("2025-01-01")
    end = frames.parse_date("2035-01-01")
    approaches = observe.close_approaches(apophis(), start, end, coarse_step_days=1.0)
    assert approaches, "expected at least one close approach"
    closest = approaches[0]
    # It should be in April 2029 and extremely close (well inside lunar distance).
    assert frames.format_jd(closest.jd, with_time=False).startswith("2029-04")
    assert closest.distance_au < 0.005
    assert closest.lunar_distances < 2.0


def test_close_approach_distance_filter():
    start = frames.parse_date("2025-01-01")
    end = frames.parse_date("2035-01-01")
    all_ca = observe.close_approaches(apophis(), start, end, coarse_step_days=2.0)
    filtered = observe.close_approaches(apophis(), start, end, coarse_step_days=2.0,
                                        max_distance_au=0.01)
    assert len(filtered) <= len(all_ca)
    assert all(ca.distance_au <= 0.01 for ca in filtered)
