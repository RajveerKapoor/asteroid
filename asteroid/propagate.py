"""Propagate orbital elements to heliocentric state vectors.

Given a set of osculating Keplerian elements valid at some epoch, this module
produces the position and velocity of the body at any other time by advancing
the mean anomaly and solving the two-body problem (see :mod:`asteroid.kepler`).

All output is in the **heliocentric ecliptic J2000** frame, with positions in AU
and velocities in AU/day — the same frame JPL's elements are given in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import kepler
from .constants import DEG2RAD, MU_SUN, RAD2DEG, AU_PER_DAY_TO_KM_S


@dataclass
class OrbitalElements:
    """Osculating Keplerian elements, stored canonically in AU and radians.

    Angles (``inc``, ``node``, ``argp``, ``M0``) are radians; ``a`` is in AU;
    ``epoch`` is a Julian Date. Build one from JPL's degree-based values with
    :meth:`from_degrees`.
    """

    a: float                 # semi-major axis (AU); negative for hyperbolic orbits
    e: float                 # eccentricity
    inc: float               # inclination (rad)
    node: float              # longitude of ascending node, Omega (rad)
    argp: float              # argument of perihelion, omega (rad)
    M0: float                # mean anomaly at epoch (rad)
    epoch: float             # epoch as Julian Date
    n: Optional[float] = None  # mean motion (rad/day); computed if None
    name: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_degrees(cls, a, e, inc_deg, node_deg, argp_deg, M0_deg, epoch,
                     n_deg=None, name="", **extra) -> "OrbitalElements":
        """Construct from JPL-style values (angles in degrees, ``n`` in deg/day)."""
        return cls(
            a=float(a),
            e=float(e),
            inc=float(inc_deg) * DEG2RAD,
            node=float(node_deg) * DEG2RAD,
            argp=float(argp_deg) * DEG2RAD,
            M0=float(M0_deg) * DEG2RAD,
            epoch=float(epoch),
            n=(float(n_deg) * DEG2RAD if n_deg is not None else None),
            name=name,
            extra=dict(extra),
        )

    @property
    def mean_motion(self) -> float:
        """Mean motion (rad/day): the published value if given, else from ``a``."""
        if self.n is not None:
            return self.n
        return kepler.mean_motion(self.a)

    @property
    def period_days(self) -> float:
        return kepler.orbital_period(self.a)

    @property
    def perihelion(self) -> float:
        return self.a * (1.0 - self.e)

    @property
    def aphelion(self) -> float:
        """Aphelion distance (AU); infinite for unbound orbits."""
        if self.e >= 1.0:
            return math.inf
        return self.a * (1.0 + self.e)

    def mean_anomaly_at(self, jd: float) -> float:
        """Mean anomaly (rad) at Julian Date ``jd``."""
        return self.M0 + self.mean_motion * (jd - self.epoch)


def rotation_matrix(node: float, inc: float, argp: float) -> np.ndarray:
    """Perifocal -> ecliptic rotation ``R_z(Omega) R_x(i) R_z(omega)``."""
    cO, sO = math.cos(node), math.sin(node)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    return np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ])


def state_vector(elements: OrbitalElements, jd: float,
                 mu: float = MU_SUN) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric ecliptic state ``(position_AU, velocity_AU_per_day)`` at ``jd``.

    Uses the unified perifocal formulation
    ``v = (mu/h) * [-sin nu, e + cos nu, 0]`` which is valid for both elliptic
    and hyperbolic orbits (``h = sqrt(mu * p)``, ``p = a(1 - e^2) > 0`` in both
    regimes).
    """
    e = elements.e
    a = elements.a

    M = elements.mean_anomaly_at(jd)
    E = kepler.mean_to_eccentric(M, e)
    nu = kepler.eccentric_to_true(E, e)
    r = kepler.radius_from_eccentric(a, e, E)

    cos_nu, sin_nu = math.cos(nu), math.sin(nu)
    pos_pf = np.array([r * cos_nu, r * sin_nu, 0.0])

    p = a * (1.0 - e * e)            # semi-latus rectum (positive in both regimes)
    h = math.sqrt(mu * p)            # specific angular momentum
    vel_pf = (mu / h) * np.array([-sin_nu, e + cos_nu, 0.0])

    rot = rotation_matrix(elements.node, elements.inc, elements.argp)
    return rot @ pos_pf, rot @ vel_pf


def position(elements: OrbitalElements, jd: float) -> np.ndarray:
    """Just the heliocentric ecliptic position (AU) at ``jd``."""
    return state_vector(elements, jd)[0]


def elements_from_state(r: np.ndarray, v: np.ndarray, epoch: float,
                        mu: float = MU_SUN, name: str = "") -> OrbitalElements:
    """Classical orbital elements from a heliocentric ecliptic state vector.

    The inverse of :func:`state_vector`: given position ``r`` (AU) and velocity
    ``v`` (AU/day) at ``epoch``, recover ``(a, e, i, Omega, omega, M0)``. Handles
    bound and hyperbolic orbits; degenerate cases (zero inclination or
    eccentricity) fall back to conventional reference directions.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    R = float(np.linalg.norm(r))
    V = float(np.linalg.norm(v))

    h_vec = np.cross(r, v)
    h = float(np.linalg.norm(h_vec))
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n = float(np.linalg.norm(n_vec))

    e_vec = ((V * V - mu / R) * r - float(np.dot(r, v)) * v) / mu
    e = float(np.linalg.norm(e_vec))

    energy = V * V / 2.0 - mu / R
    a = math.inf if abs(e - 1.0) < 1e-12 else -mu / (2.0 * energy)

    inc = math.acos(_clip(h_vec[2] / h))

    if n > 1e-12:
        node = math.acos(_clip(n_vec[0] / n))
        if n_vec[1] < 0:
            node = 2.0 * math.pi - node
    else:
        node = 0.0  # equatorial orbit: ascending node undefined -> reference x

    if e > 1e-12 and n > 1e-12:
        argp = math.acos(_clip(float(np.dot(n_vec, e_vec)) / (n * e)))
        if e_vec[2] < 0:
            argp = 2.0 * math.pi - argp
    elif e > 1e-12:
        argp = math.acos(_clip(e_vec[0] / e))      # equatorial: from x-axis
        if e_vec[1] < 0:
            argp = 2.0 * math.pi - argp
    else:
        argp = 0.0  # circular orbit: perihelion undefined

    if e > 1e-12:
        nu = math.acos(_clip(float(np.dot(e_vec, r)) / (e * R)))
        if float(np.dot(r, v)) < 0:
            nu = 2.0 * math.pi - nu
    elif n > 1e-12:
        nu = math.acos(_clip(float(np.dot(n_vec, r)) / (n * R)))
        if r[2] < 0:
            nu = 2.0 * math.pi - nu
    else:
        nu = math.acos(_clip(r[0] / R))

    big_e = kepler.true_to_eccentric(nu, e)
    M0 = kepler.eccentric_to_mean(big_e, e)

    return OrbitalElements(
        a=a, e=e, inc=inc, node=node, argp=argp, M0=M0,
        epoch=epoch, n=kepler.mean_motion(a, mu), name=name,
    )


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


@dataclass
class StateSample:
    """One row of an ephemeris."""

    jd: float
    position: np.ndarray          # heliocentric ecliptic (AU)
    velocity: np.ndarray          # heliocentric ecliptic (AU/day)

    @property
    def r(self) -> float:
        """Heliocentric distance (AU)."""
        return float(np.linalg.norm(self.position))

    @property
    def speed_km_s(self) -> float:
        return float(np.linalg.norm(self.velocity)) * AU_PER_DAY_TO_KM_S


def ephemeris(elements: OrbitalElements, jd_start: float, jd_end: float,
              step_days: float) -> list:
    """Sample the trajectory from ``jd_start`` to ``jd_end`` every ``step_days``."""
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    samples = []
    n_steps = int(math.floor((jd_end - jd_start) / step_days + 1e-9))
    for k in range(n_steps + 1):
        jd = jd_start + k * step_days
        pos, vel = state_vector(elements, jd)
        samples.append(StateSample(jd=jd, position=pos, velocity=vel))
    return samples


def orbit_path(elements: OrbitalElements, n_points: int = 360) -> np.ndarray:
    """Sample the full orbit shape as an ``(n_points, 3)`` array of positions.

    For bound orbits this traces one closed revolution (uniform in eccentric
    anomaly). For hyperbolic orbits it sweeps the true anomaly between the
    asymptotes, giving the visible arc.
    """
    a, e = elements.a, elements.e
    rot = rotation_matrix(elements.node, elements.inc, elements.argp)
    pts = np.empty((n_points, 3))

    if e < 1.0:
        for k in range(n_points):
            big_e = -math.pi + 2.0 * math.pi * k / (n_points - 1)
            nu = kepler.eccentric_to_true(big_e, e)
            r = kepler.radius_from_eccentric(a, e, big_e)
            pts[k] = rot @ np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    else:
        nu_inf = math.acos(-1.0 / e)        # asymptote true anomaly
        nu_max = 0.98 * nu_inf              # stay just inside the asymptotes
        p = a * (1.0 - e * e)
        for k in range(n_points):
            nu = -nu_max + 2.0 * nu_max * k / (n_points - 1)
            r = p / (1.0 + e * math.cos(nu))
            pts[k] = rot @ np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    return pts


def time_of_perihelion(elements: OrbitalElements, after_jd: float) -> float:
    """Julian Date of the first perihelion passage at or after ``after_jd``.

    Perihelion is where the mean anomaly is a multiple of 2*pi.
    """
    n = elements.mean_motion
    # Mean anomaly at after_jd, then advance to the next multiple of 2*pi.
    M = elements.mean_anomaly_at(after_jd)
    frac = M % (2.0 * math.pi)
    delta_M = (2.0 * math.pi - frac) % (2.0 * math.pi)
    return after_jd + delta_M / n
