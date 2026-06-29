"""Tests for the two-body core: Kepler's equation and anomaly conversions."""

import math

import pytest

from asteroid import kepler
from asteroid.constants import GAUSS_K


# A spread of eccentricities covering circular, typical, and near-parabolic.
ELLIPTIC_E = [0.0, 0.0167, 0.2, 0.5, 0.8, 0.95, 0.99]
HYPERBOLIC_E = [1.05, 1.5, 3.0]
MEAN_ANOMALIES = [(-3.0 + 0.37 * k) for k in range(17)]  # span ~[-3, 3] rad


@pytest.mark.parametrize("e", ELLIPTIC_E)
@pytest.mark.parametrize("m", MEAN_ANOMALIES)
def test_elliptic_kepler_residual(e, m):
    """The solved E must satisfy Kepler's equation to machine precision."""
    big_e = kepler.solve_kepler_elliptic(m, e)
    residual = big_e - e * math.sin(big_e) - kepler.wrap_angle(m)
    assert abs(residual) < 1e-11


@pytest.mark.parametrize("e", ELLIPTIC_E)
@pytest.mark.parametrize("m", MEAN_ANOMALIES)
def test_elliptic_round_trip(e, m):
    """M -> E -> M should return the (wrapped) original mean anomaly."""
    big_e = kepler.mean_to_eccentric(m, e)
    m_back = kepler.eccentric_to_mean(big_e, e)
    assert abs(kepler.wrap_angle(m_back - m)) < 1e-10


def test_circular_orbit_E_equals_M():
    """With e=0 the eccentric anomaly equals the mean anomaly."""
    for m in MEAN_ANOMALIES:
        assert kepler.solve_kepler_elliptic(m, 0.0) == pytest.approx(
            kepler.wrap_angle(m), abs=1e-12
        )


@pytest.mark.parametrize("e", ELLIPTIC_E)
@pytest.mark.parametrize("m", MEAN_ANOMALIES)
def test_true_anomaly_round_trip_elliptic(e, m):
    """E -> nu -> E should be the identity."""
    big_e = kepler.solve_kepler_elliptic(m, e)
    nu = kepler.eccentric_to_true(big_e, e)
    e_back = kepler.true_to_eccentric(nu, e)
    assert abs(kepler.wrap_angle(e_back - big_e)) < 1e-9


@pytest.mark.parametrize("e", HYPERBOLIC_E)
@pytest.mark.parametrize("m", MEAN_ANOMALIES)
def test_hyperbolic_kepler_residual(e, m):
    """The hyperbolic solver must satisfy M = e*sinh(H) - H."""
    h = kepler.solve_kepler_hyperbolic(m, e)
    residual = e * math.sinh(h) - h - m
    assert abs(residual) < 1e-9


@pytest.mark.parametrize("e", HYPERBOLIC_E)
@pytest.mark.parametrize("m", MEAN_ANOMALIES)
def test_hyperbolic_round_trip(e, m):
    """M -> H -> M and H -> nu -> H both round-trip for hyperbolic orbits."""
    h = kepler.mean_to_eccentric(m, e)
    assert abs(kepler.eccentric_to_mean(h, e) - m) < 1e-8
    nu = kepler.eccentric_to_true(h, e)
    assert abs(kepler.true_to_eccentric(nu, e) - h) < 1e-8


def test_perihelion_is_closest_point():
    """At nu=0 (perihelion) r = a(1-e); at nu=pi (aphelion) r = a(1+e)."""
    a, e = 2.5, 0.3
    r_peri = kepler.radius_from_eccentric(a, e, 0.0)
    r_apo = kepler.radius_from_eccentric(a, e, math.pi)
    assert r_peri == pytest.approx(a * (1 - e), rel=1e-12)
    assert r_apo == pytest.approx(a * (1 + e), rel=1e-12)


def test_mean_motion_and_period():
    """At a=1 AU the mean motion equals Gauss's constant; period ~ a sidereal year."""
    assert kepler.mean_motion(1.0) == pytest.approx(GAUSS_K, rel=1e-12)
    # 2*pi / k = 365.2568983... days
    assert kepler.orbital_period(1.0) == pytest.approx(365.2568983, abs=1e-3)
    # Kepler's third law: period scales as a^1.5.
    assert kepler.orbital_period(4.0) == pytest.approx(
        kepler.orbital_period(1.0) * 8.0, rel=1e-12
    )


def test_invalid_eccentricity_raises():
    with pytest.raises(ValueError):
        kepler.solve_kepler_elliptic(1.0, 1.5)
    with pytest.raises(ValueError):
        kepler.solve_kepler_hyperbolic(1.0, 0.5)
