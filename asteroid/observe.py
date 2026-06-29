"""Observable quantities: where the body is in the sky, how bright, how close.

Given an asteroid's heliocentric state and Earth's, this module derives the
things an observer actually cares about:

* **delta** — distance from Earth (AU)
* **RA / Dec** — geocentric sky position (equatorial J2000)
* **ecliptic lon/lat** — heliocentric position angle in the solar system
* **phase angle** and **solar elongation** — viewing geometry
* **apparent magnitude** — via the IAU H-G photometric system

It also includes a :func:`close_approaches` scanner that finds and refines the
local minima of the Earth-distance over a date range — e.g. Apophis' 2029 flyby.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import frames
from .bodies import earth_position
from .constants import AU_KM, AU_PER_DAY_TO_KM_S
from .propagate import OrbitalElements, state_vector, position


@dataclass
class Observation:
    """A complete snapshot of a body as seen at one instant."""

    jd: float
    r_helio: float                 # asteroid-Sun distance (AU)
    delta: float                   # asteroid-Earth distance (AU)
    helio_lon: float               # heliocentric ecliptic longitude (deg)
    helio_lat: float               # heliocentric ecliptic latitude (deg)
    ra: float                      # geocentric right ascension (deg)
    dec: float                     # geocentric declination (deg)
    elongation: float              # solar elongation from Earth (deg)
    phase_angle: float             # Sun-asteroid-Earth angle (deg)
    speed_helio_km_s: float        # heliocentric speed (km/s)
    magnitude: Optional[float]     # apparent V magnitude (None if H unknown)
    position_helio: np.ndarray     # heliocentric ecliptic position (AU)

    @property
    def delta_km(self) -> float:
        return self.delta * AU_KM

    @property
    def lunar_distances(self) -> float:
        """Earth distance expressed in mean Earth-Moon distances (384,400 km)."""
        return self.delta_km / 384_400.0


def phase_integral_magnitude(H: float, r: float, delta: float, alpha_rad: float,
                             G: float = 0.15) -> float:
    """Apparent magnitude via the IAU two-parameter H-G system.

    ``V = H + 5*log10(r*delta) - 2.5*log10((1-G)*phi1 + G*phi2)``.
    """
    tan_half = math.tan(alpha_rad / 2.0)
    phi1 = math.exp(-3.33 * tan_half ** 0.63)
    phi2 = math.exp(-1.87 * tan_half ** 1.22)
    phase = (1.0 - G) * phi1 + G * phi2
    phase = max(phase, 1e-12)
    return H + 5.0 * math.log10(r * delta) - 2.5 * math.log10(phase)


def observe(elements: OrbitalElements, jd: float,
            H: Optional[float] = None, G: float = 0.15) -> Observation:
    """Build a full :class:`Observation` for ``elements`` at ``jd``."""
    ast, vel = state_vector(elements, jd)
    earth = earth_position(jd)
    geo = ast - earth                      # asteroid as seen from Earth (ecliptic)

    r = float(np.linalg.norm(ast))
    delta = float(np.linalg.norm(geo))
    sun_earth = float(np.linalg.norm(earth))

    # Heliocentric ecliptic angles (orbital position).
    _, helio_lon, helio_lat = frames.cartesian_to_spherical(ast)

    # Geocentric equatorial sky position.
    ra, dec = frames.equatorial_to_radec(frames.ecliptic_to_equatorial(geo))

    # Phase angle: Sun-asteroid-Earth.
    to_sun = -ast
    to_earth = earth - ast
    phase = _angle_between(to_sun, to_earth)

    # Solar elongation: Sun-Earth-asteroid.
    to_sun_from_earth = -earth
    elong = _angle_between(to_sun_from_earth, geo)

    mag = None
    if H is not None:
        mag = phase_integral_magnitude(H, r, delta, math.radians(phase), G)

    return Observation(
        jd=jd, r_helio=r, delta=delta,
        helio_lon=helio_lon, helio_lat=helio_lat,
        ra=ra, dec=dec, elongation=elong, phase_angle=phase,
        speed_helio_km_s=float(np.linalg.norm(vel)) * AU_PER_DAY_TO_KM_S,
        magnitude=mag, position_helio=ast,
    )


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle (degrees) between two vectors, numerically safe."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    c = float(np.dot(u, v) / (nu * nv))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# --------------------------------------------------------------------------- #
# Close-approach scanner
# --------------------------------------------------------------------------- #
@dataclass
class CloseApproach:
    jd: float
    distance_au: float

    @property
    def distance_km(self) -> float:
        return self.distance_au * AU_KM

    @property
    def lunar_distances(self) -> float:
        return self.distance_km / 384_400.0


def geocentric_distance(elements: OrbitalElements, jd: float) -> float:
    """Asteroid-Earth distance (AU) at ``jd``."""
    return float(np.linalg.norm(position(elements, jd) - earth_position(jd)))


def _golden_section_min(f, lo: float, hi: float, tol: float = 1e-5):
    """Minimise a unimodal ``f`` on ``[lo, hi]``; returns ``(x_min, f_min)``."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0       # 1/phi
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0      # 1/phi^2
    a, b = lo, hi
    h = b - a
    c, d = a + invphi2 * h, a + invphi * h
    fc, fd = f(c), f(d)
    while h > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            h = b - a
            c = a + invphi2 * h
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            h = b - a
            d = a + invphi * h
            fd = f(d)
    x = (a + b) / 2.0
    return x, f(x)


def close_approaches(elements: OrbitalElements, jd_start: float, jd_end: float,
                     coarse_step_days: float = 1.0,
                     max_distance_au: Optional[float] = None) -> list:
    """Find Earth close approaches between ``jd_start`` and ``jd_end``.

    The Earth-distance is sampled coarsely; each local minimum is bracketed and
    refined to ~second precision with a golden-section search. Results are sorted
    closest-first.
    """
    if coarse_step_days <= 0:
        raise ValueError("coarse_step_days must be positive")

    def dist(jd):
        return geocentric_distance(elements, jd)

    n = int(math.ceil((jd_end - jd_start) / coarse_step_days))
    times = [jd_start + k * coarse_step_days for k in range(n + 1)]
    if times[-1] < jd_end:
        times.append(jd_end)
    dists = [dist(t) for t in times]

    approaches = []
    for i in range(1, len(times) - 1):
        if dists[i] <= dists[i - 1] and dists[i] <= dists[i + 1]:
            jd_min, d_min = _golden_section_min(dist, times[i - 1], times[i + 1])
            approaches.append(CloseApproach(jd=jd_min, distance_au=d_min))

    # Merge near-duplicate minima from adjacent brackets.
    approaches.sort(key=lambda ca: ca.jd)
    merged = []
    for ca in approaches:
        if merged and abs(ca.jd - merged[-1].jd) < coarse_step_days:
            if ca.distance_au < merged[-1].distance_au:
                merged[-1] = ca
        else:
            merged.append(ca)

    if max_distance_au is not None:
        merged = [ca for ca in merged if ca.distance_au <= max_distance_au]
    merged.sort(key=lambda ca: ca.distance_au)
    return merged
