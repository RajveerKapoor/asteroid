"""Tests for propagation and planetary positions.

These check the propagator against the conservation laws it must obey
(vis-viva energy, angular momentum, periodicity) and sanity-check Earth's
position against known astronomy.
"""

import math

import numpy as np
import pytest

from asteroid import bodies, frames
from asteroid.constants import MU_SUN
from asteroid.propagate import OrbitalElements, state_vector, ephemeris, orbit_path


def make_elements(a=2.0, e=0.3, inc=10.0, node=80.0, argp=60.0, M0=0.0, epoch=2451545.0):
    return OrbitalElements.from_degrees(a, e, inc, node, argp, M0, epoch)


def test_perihelion_distance_and_speed():
    """At M0=0 the body sits at perihelion: r = a(1-e)."""
    a, e = 2.0, 0.3
    el = make_elements(a=a, e=e, M0=0.0)
    pos, vel = state_vector(el, el.epoch)
    r = np.linalg.norm(pos)
    assert r == pytest.approx(a * (1 - e), rel=1e-10)
    # Vis-viva at perihelion.
    v2 = MU_SUN * (2.0 / r - 1.0 / a)
    assert np.dot(vel, vel) == pytest.approx(v2, rel=1e-10)


@pytest.mark.parametrize("M0", [0.0, 35.0, 90.0, 170.0, 260.0])
def test_vis_viva_holds_everywhere(M0):
    """Specific orbital energy implies speed^2 = mu(2/r - 1/a) at every point."""
    a = 2.5
    el = make_elements(a=a, e=0.4, M0=M0)
    for djd in [0.0, 50.0, 123.4, 400.0]:
        pos, vel = state_vector(el, el.epoch + djd)
        r = np.linalg.norm(pos)
        assert np.dot(vel, vel) == pytest.approx(MU_SUN * (2.0 / r - 1.0 / a), rel=1e-9)


def test_angular_momentum_conserved():
    """|r x v| is constant along the orbit and equals sqrt(mu * p)."""
    a, e = 3.0, 0.5
    el = make_elements(a=a, e=e)
    p = a * (1 - e * e)
    h_expected = math.sqrt(MU_SUN * p)
    for djd in [0.0, 100.0, 500.0, 900.0]:
        pos, vel = state_vector(el, el.epoch + djd)
        h = np.linalg.norm(np.cross(pos, vel))
        assert h == pytest.approx(h_expected, rel=1e-9)


def test_periodicity():
    """After exactly one period the state returns to where it started."""
    el = make_elements(a=2.0, e=0.25)
    pos0, vel0 = state_vector(el, el.epoch)
    pos1, vel1 = state_vector(el, el.epoch + el.period_days)
    assert np.allclose(pos0, pos1, atol=1e-9)
    assert np.allclose(vel0, vel1, atol=1e-11)


def test_inclination_controls_z_extent():
    """A zero-inclination orbit stays in the ecliptic plane (z == 0)."""
    flat = make_elements(inc=0.0)
    for djd in [0.0, 60.0, 250.0]:
        pos, _ = state_vector(flat, flat.epoch + djd)
        assert abs(pos[2]) < 1e-12


def test_ephemeris_length_and_endpoints():
    el = make_elements()
    samples = ephemeris(el, el.epoch, el.epoch + 100.0, 10.0)
    assert len(samples) == 11
    assert samples[0].jd == pytest.approx(el.epoch)
    assert samples[-1].jd == pytest.approx(el.epoch + 100.0)


def test_orbit_path_closes_for_bound_orbit():
    el = make_elements(a=2.0, e=0.4)
    path = orbit_path(el, n_points=720)
    # First and last sampled points coincide (full revolution).
    assert np.allclose(path[0], path[-1], atol=1e-9)
    # Every point lies between perihelion and aphelion.
    radii = np.linalg.norm(path, axis=1)
    assert radii.min() == pytest.approx(el.perihelion, rel=1e-3)
    assert radii.max() == pytest.approx(el.aphelion, rel=1e-3)


# --------------------------------------------------------------------------- #
# Earth / planets
# --------------------------------------------------------------------------- #
def test_earth_distance_range():
    """Earth's heliocentric distance always sits between perihelion and aphelion."""
    for date in ["2000-01-03", "2020-07-04", "2026-06-29", "2029-04-13"]:
        jd = frames.parse_date(date)
        r = np.linalg.norm(bodies.earth_position(jd))
        assert 0.98 < r < 1.02


def test_earth_longitude_at_summer_solstice():
    """Near the June solstice Earth's heliocentric ecliptic longitude is ~270 deg."""
    jd = frames.parse_date("2000-06-21")
    pos = bodies.earth_position(jd)
    _, lon, lat = frames.cartesian_to_spherical(pos)
    assert lon == pytest.approx(270.0, abs=2.0)
    assert abs(lat) < 0.01  # Earth defines the ecliptic, so latitude ~ 0


def test_all_planets_have_sane_distances():
    """Each planet must fall within its own perihelion-aphelion shell."""
    jd = frames.parse_date("2026-06-29")
    for name in bodies.PLANET_ORDER:
        el = bodies.planet_elements_at(name, jd)
        r = np.linalg.norm(bodies.planet_position(name, jd))
        assert el.perihelion - 0.02 <= r <= el.aphelion + 0.02
