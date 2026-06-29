"""The two-body core: Kepler's equation and anomaly conversions.

Everything here is pure math on scalars in **radians**, with no I/O and no
dependencies beyond the standard library. These are the functions that turn a
mean anomaly (which advances linearly in time) into a real position on the
orbit, and they are exercised hard by ``tests/test_kepler.py``.

Three "anomalies" describe where a body is on its orbit:

* **Mean anomaly** ``M`` — a fictitious angle that grows linearly with time.
* **Eccentric anomaly** ``E`` (elliptic) / ``H`` (hyperbolic) — a geometric
  angle obtained by solving Kepler's equation.
* **True anomaly** ``nu`` — the actual angle from perihelion to the body.

Kepler's equation links the first two and has no closed-form solution, so we
solve it with Newton–Raphson iteration.
"""

from __future__ import annotations

import math

from .constants import MU_SUN, TWO_PI


def wrap_angle(angle: float) -> float:
    """Wrap an angle (radians) into ``[-pi, pi]`` for numerical stability."""
    return (angle + math.pi) % TWO_PI - math.pi


def solve_kepler_elliptic(mean_anomaly: float, e: float, tol: float = 1e-13,
                          max_iter: int = 100) -> float:
    """Solve ``M = E - e*sin(E)`` for the eccentric anomaly ``E`` (radians).

    Newton–Raphson with a seed chosen for robustness at high eccentricity.
    Converges quadratically for all ``0 <= e < 1``.
    """
    if not 0.0 <= e < 1.0:
        raise ValueError(f"elliptic solver requires 0 <= e < 1, got e={e}")

    m = wrap_angle(mean_anomaly)
    # A constant seed of pi is safe near e -> 1 where E = M is a poor guess.
    e_anom = m if e < 0.8 else math.copysign(math.pi, m) if m != 0 else math.pi
    for _ in range(max_iter):
        delta = (e_anom - e * math.sin(e_anom) - m) / (1.0 - e * math.cos(e_anom))
        e_anom -= delta
        if abs(delta) < tol:
            break
    return e_anom


def solve_kepler_hyperbolic(mean_anomaly: float, e: float, tol: float = 1e-13,
                            max_iter: int = 200) -> float:
    """Solve ``M = e*sinh(H) - H`` for the hyperbolic anomaly ``H`` (radians).

    Used for unbound orbits (``e > 1``) such as interstellar objects.
    """
    if e <= 1.0:
        raise ValueError(f"hyperbolic solver requires e > 1, got e={e}")

    m = mean_anomaly
    # asinh(M/e) is a well-behaved seed across the whole range of M.
    h = math.asinh(m / e) if e != 0 else m
    for _ in range(max_iter):
        f = e * math.sinh(h) - h - m
        fp = e * math.cosh(h) - 1.0
        delta = f / fp
        h -= delta
        if abs(delta) < tol:
            break
    return h


def mean_to_eccentric(mean_anomaly: float, e: float) -> float:
    """Solve Kepler's equation, dispatching on eccentricity.

    Returns the eccentric anomaly ``E`` for ``e < 1`` or the hyperbolic
    anomaly ``H`` for ``e > 1``.
    """
    if e < 1.0:
        return solve_kepler_elliptic(mean_anomaly, e)
    return solve_kepler_hyperbolic(mean_anomaly, e)


def eccentric_to_mean(eccentric_anomaly: float, e: float) -> float:
    """Inverse of Kepler's equation (cheap, closed form). Useful for testing."""
    if e < 1.0:
        return eccentric_anomaly - e * math.sin(eccentric_anomaly)
    return e * math.sinh(eccentric_anomaly) - eccentric_anomaly


def eccentric_to_true(eccentric_anomaly: float, e: float) -> float:
    """True anomaly ``nu`` (radians) from eccentric/hyperbolic anomaly."""
    if e < 1.0:
        half = eccentric_anomaly / 2.0
        return 2.0 * math.atan2(
            math.sqrt(1.0 + e) * math.sin(half),
            math.sqrt(1.0 - e) * math.cos(half),
        )
    half = eccentric_anomaly / 2.0
    return 2.0 * math.atan2(
        math.sqrt(e + 1.0) * math.sinh(half),
        math.sqrt(e - 1.0) * math.cosh(half),
    )


def true_to_eccentric(true_anomaly: float, e: float) -> float:
    """Eccentric/hyperbolic anomaly from the true anomaly ``nu`` (radians)."""
    if e < 1.0:
        half = true_anomaly / 2.0
        return 2.0 * math.atan2(
            math.sqrt(1.0 - e) * math.sin(half),
            math.sqrt(1.0 + e) * math.cos(half),
        )
    half = true_anomaly / 2.0
    return 2.0 * math.atanh(
        math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(half)
    )


def radius_from_eccentric(a: float, e: float, eccentric_anomaly: float) -> float:
    """Heliocentric distance ``r`` (AU) from the eccentric/hyperbolic anomaly.

    Works for both regimes: for hyperbolas ``a`` is negative so the formula
    ``r = a*(1 - e*cosh(H))`` returns a positive distance.
    """
    if e < 1.0:
        return a * (1.0 - e * math.cos(eccentric_anomaly))
    return a * (1.0 - e * math.cosh(eccentric_anomaly))


def mean_motion(a: float, mu: float = MU_SUN) -> float:
    """Mean motion ``n`` (radians/day) from the semi-major axis.

    ``n = sqrt(mu / |a|^3)``. Uses ``|a|`` so hyperbolic orbits (``a < 0``)
    return a positive rate.
    """
    return math.sqrt(mu / abs(a) ** 3)


def orbital_period(a: float, mu: float = MU_SUN) -> float:
    """Orbital period (days) for a bound orbit; ``inf`` for ``a <= 0``."""
    if a <= 0:
        return math.inf
    return TWO_PI / mean_motion(a, mu)
