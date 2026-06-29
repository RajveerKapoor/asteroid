"""Tests for the N-body numerical propagator (asteroid.nbody).

The integrator is validated *offline* against things it must obey regardless of
any external data:

* with the planets switched off it must reproduce the analytic two-body solution
  to high precision (this proves the Dormand-Prince step is correct);
* with the planets off, specific orbital energy is conserved;
* forward-then-backward integration returns to the start (time reversibility);
* turning the planets on perturbs the state by a small but non-zero amount;
* ``ephemeris`` agrees with one-off ``propagate_elements`` calls.
"""

import math

import numpy as np
import pytest

from asteroid import nbody
from asteroid.constants import MU_SUN
from asteroid.propagate import OrbitalElements, state_vector


def make_elements(a=2.2, e=0.25, inc=12.0, node=70.0, argp=50.0, M0=20.0,
                  epoch=2451545.0):
    return OrbitalElements.from_degrees(a, e, inc, node, argp, M0, epoch)


def test_acceleration_is_sun_dominated_at_1au():
    """At 1 AU the Sun term dominates and points back toward the Sun."""
    r = np.array([1.0, 0.0, 0.0])
    a = nbody.acceleration(r, 2451545.0)
    assert a[0] < 0.0                                  # pulled toward the Sun
    assert np.linalg.norm(a) == pytest.approx(MU_SUN, rel=5e-3)  # ~ -mu/r^2


def test_planet_gm_positive_and_jupiter_largest():
    assert all(v > 0 for v in nbody.PLANET_GM.values())
    assert max(nbody.PLANET_GM, key=nbody.PLANET_GM.get) == "Jupiter"


def test_at_epoch_returns_epoch_state_exactly():
    el = make_elements()
    r, v = nbody.propagate_elements(el, el.epoch)
    r0, v0 = state_vector(el, el.epoch)
    assert np.allclose(r, r0, atol=0)
    assert np.allclose(v, v0, atol=0)


@pytest.mark.parametrize("dt", [50.0, 200.0, -120.0])
def test_two_body_limit_matches_analytic(dt):
    """Planets off => integration must equal the exact Kepler propagation."""
    el = make_elements()
    r0, v0 = state_vector(el, el.epoch)
    r_num, v_num = nbody.propagate_state(
        r0, v0, el.epoch, el.epoch + dt, perturbers=(), rtol=1e-12, atol=1e-14)
    r_exact, v_exact = state_vector(el, el.epoch + dt)
    assert np.linalg.norm(r_num - r_exact) < 1e-9      # AU
    assert np.linalg.norm(v_num - v_exact) < 1e-11     # AU/day


def test_energy_conserved_without_planets():
    el = make_elements(e=0.4)
    r0, v0 = state_vector(el, el.epoch)
    energy0 = 0.5 * (v0 @ v0) - MU_SUN / np.linalg.norm(r0)
    r, v = nbody.propagate_state(r0, v0, el.epoch, el.epoch + 365.0,
                                 perturbers=(), rtol=1e-12, atol=1e-14)
    energy1 = 0.5 * (v @ v) - MU_SUN / np.linalg.norm(r)
    assert energy1 == pytest.approx(energy0, rel=1e-9)


def test_time_reversibility_with_planets():
    """Integrate forward a year then back; we should return to the start."""
    el = make_elements()
    r0, v0 = state_vector(el, el.epoch)
    r1, v1 = nbody.propagate_state(r0, v0, el.epoch, el.epoch + 365.0)
    r2, v2 = nbody.propagate_state(r1, v1, el.epoch + 365.0, el.epoch)
    assert np.linalg.norm(r2 - r0) < 1e-7              # AU
    assert np.linalg.norm(v2 - v0) < 1e-9


def test_planets_perturb_the_orbit():
    """The full force model must differ from two-body by a small, real amount."""
    el = make_elements()
    r_nb, _ = nbody.propagate_elements(el, el.epoch + 365.0)
    r_2b, _ = state_vector(el, el.epoch + 365.0)
    sep = np.linalg.norm(r_nb - r_2b)
    assert 1e-7 < sep < 1e-1                            # AU: nonzero but modest


def test_ephemeris_matches_pointwise():
    el = make_elements()
    jds = [el.epoch - 100.0, el.epoch + 30.0, el.epoch + 400.0]
    rows = nbody.ephemeris(el, jds)
    assert [t for t, _, _ in rows] == sorted(jds)
    for t, r, v in rows:
        r1, v1 = nbody.propagate_elements(el, t)
        assert np.linalg.norm(r - r1) < 1e-8
        assert np.linalg.norm(v - v1) < 1e-10
