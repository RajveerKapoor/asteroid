"""Positions of the major planets (and Earth) via the JPL approximate ephemeris.

Computing an asteroid's distance from Earth, its phase angle, solar elongation
and apparent magnitude all require Earth's heliocentric position. We get it from
Standish's "Keplerian Elements for Approximate Positions of the Major Planets"
(JPL), which gives each planet's elements at J2000 plus linear rates per century,
valid 1800-2050 to roughly arc-minute accuracy.

Each planet's time-varying elements are evaluated at the requested instant and
then fed straight through the same :func:`asteroid.propagate.state_vector`
two-body machinery used for asteroids, so there is a single rotation/solver path.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import J2000_JD, DAYS_PER_CENTURY, EARTH_RADIUS_AU
from .propagate import OrbitalElements, state_vector

# Moon mass as a fraction of the Earth-Moon system, for the barycentre offset.
_MOON_MASS_FRACTION = 1.0 / 82.30056

# name -> (a0, adot, e0, edot, I0, Idot, L0, Ldot, varpi0, varpidot, Om0, Omdot)
# a in AU (rate AU/century); all angles in degrees (rates deg/century).
# Source: https://ssd.jpl.nasa.gov/planets/approx_pos.html (Table 1, 1800-2050).
PLANET_ELEMENTS = {
    "Mercury": (0.38709927, 0.00000037, 0.20563593, 0.00001906,
                7.00497902, -0.00594749, 252.25032350, 149472.67411175,
                77.45779628, 0.16047689, 48.33076593, -0.12534081),
    "Venus": (0.72333566, 0.00000390, 0.00677672, -0.00004107,
              3.39467605, -0.00078890, 181.97909950, 58517.81538729,
              131.60246718, 0.00268329, 76.67984255, -0.27769418),
    "Earth": (1.00000261, 0.00000562, 0.01671123, -0.00004392,
              -0.00001531, -0.01294668, 100.46457166, 35999.37244981,
              102.93768193, 0.32327364, 0.0, 0.0),
    "Mars": (1.52371034, 0.00001847, 0.09339410, 0.00007882,
             1.84969142, -0.00813131, -4.55343205, 19140.30268499,
             -23.94362959, 0.44441088, 49.55953891, -0.29257343),
    "Jupiter": (5.20288700, -0.00011607, 0.04838624, -0.00013253,
                1.30439695, -0.00183714, 34.39644051, 3034.74612775,
                14.72847983, 0.21252668, 100.47390909, 0.20469106),
    "Saturn": (9.53667594, -0.00125060, 0.05386179, -0.00050991,
               2.48599187, 0.00193609, 49.95424423, 1222.49362201,
               92.59887831, -0.41897216, 113.66242448, -0.28867794),
    "Uranus": (19.18916464, -0.00196176, 0.04725744, -0.00004397,
               0.77263783, -0.00242939, 313.23810451, 428.48202785,
               170.95427630, 0.40805281, 74.01692503, 0.04240589),
    "Neptune": (30.06992276, 0.00026291, 0.00859048, 0.00005105,
                1.77004347, 0.00035372, -55.12002969, 218.45945325,
                44.96476227, -0.32241464, 131.78422574, -0.00508664),
}

PLANET_ORDER = list(PLANET_ELEMENTS.keys())


def planet_elements_at(name: str, jd: float) -> OrbitalElements:
    """Osculating elements of a planet at Julian Date ``jd``.

    Applies the linear secular rates, then converts the longitude of perihelion
    and mean longitude into the argument of perihelion and mean anomaly that the
    two-body propagator expects.
    """
    try:
        (a0, adot, e0, edot, i0, idot, l0, ldot,
         w0, wdot, om0, omdot) = PLANET_ELEMENTS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown planet {name!r}; choose from {PLANET_ORDER}") from exc

    t = (jd - J2000_JD) / DAYS_PER_CENTURY
    a = a0 + adot * t
    e = e0 + edot * t
    inc = i0 + idot * t
    mean_long = l0 + ldot * t          # mean longitude L
    lon_peri = w0 + wdot * t           # longitude of perihelion varpi
    node = om0 + omdot * t             # longitude of ascending node Omega

    argp = lon_peri - node             # argument of perihelion omega
    mean_anom = mean_long - lon_peri   # mean anomaly M
    # Reduce to [-180, 180] deg for a good Kepler seed.
    mean_anom = (mean_anom + 180.0) % 360.0 - 180.0

    return OrbitalElements.from_degrees(
        a=a, e=e, inc_deg=inc, node_deg=node, argp_deg=argp,
        M0_deg=mean_anom, epoch=jd, name=name,
    )


def planet_state(name: str, jd: float) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric ecliptic ``(position_AU, velocity_AU_per_day)`` of a planet."""
    return state_vector(planet_elements_at(name, jd), jd)


def planet_position(name: str, jd: float) -> np.ndarray:
    return planet_state(name, jd)[0]


def earth_position(jd: float) -> np.ndarray:
    """Heliocentric ecliptic position of Earth (Earth-Moon barycentre) in AU."""
    return planet_position("Earth", jd)


def earth_state(jd: float) -> tuple[np.ndarray, np.ndarray]:
    return planet_state("Earth", jd)


def moon_position_geocentric(jd: float) -> np.ndarray:
    """Geocentric ecliptic position of the Moon (AU), low precision.

    Compact lunar theory (after Schlyter): the two-body lunar orbit plus the
    dozen largest periodic perturbations. Good to ~1-2 arc-minutes — far more
    than enough to locate the Earth-Moon barycentre to a few tens of km.
    """
    d = jd - 2451543.5  # days since 2000 Jan 0.0
    r = math.radians

    N = 125.1228 - 0.0529538083 * d     # ascending node
    inc = 5.1454                        # inclination
    w = 318.0634 + 0.1643573223 * d     # argument of perihelion
    a = 60.2666                         # semi-major axis (Earth radii)
    e = 0.054900
    M = 115.3654 + 13.0649929509 * d    # mean anomaly
    Ms = 356.0470 + 0.9856002585 * d    # Sun's mean anomaly
    ws = 282.9404 + 4.70935e-5 * d      # Sun's perihelion
    Lm = N + w + M                      # Moon's mean longitude
    Ls = ws + Ms                        # Sun's mean longitude
    D = Lm - Ls                         # mean elongation
    F = Lm - N                          # argument of latitude

    # Eccentric anomaly (degrees), a couple of Newton iterations.
    E = M + (180.0 / math.pi) * e * math.sin(r(M)) * (1.0 + e * math.cos(r(M)))
    for _ in range(2):
        E -= (E - (180.0 / math.pi) * e * math.sin(r(E)) - M) / (1.0 - e * math.cos(r(E)))

    xv = a * (math.cos(r(E)) - e)
    yv = a * math.sqrt(1.0 - e * e) * math.sin(r(E))
    rad_dist = math.hypot(xv, yv)
    v = math.degrees(math.atan2(yv, xv))

    vw = r(v + w)
    xh = rad_dist * (math.cos(r(N)) * math.cos(vw) - math.sin(r(N)) * math.sin(vw) * math.cos(r(inc)))
    yh = rad_dist * (math.sin(r(N)) * math.cos(vw) + math.cos(r(N)) * math.sin(vw) * math.cos(r(inc)))
    zh = rad_dist * (math.sin(vw) * math.sin(r(inc)))

    lon = math.degrees(math.atan2(yh, xh))
    lat = math.degrees(math.atan2(zh, math.hypot(xh, yh)))

    lon += (-1.274 * math.sin(r(M - 2 * D)) + 0.658 * math.sin(r(2 * D))
            - 0.186 * math.sin(r(Ms)) - 0.059 * math.sin(r(2 * M - 2 * D))
            - 0.057 * math.sin(r(M - 2 * D + Ms)) + 0.053 * math.sin(r(M + 2 * D))
            + 0.046 * math.sin(r(2 * D - Ms)) + 0.041 * math.sin(r(M - Ms))
            - 0.035 * math.sin(r(D)) - 0.031 * math.sin(r(M + Ms))
            - 0.015 * math.sin(r(2 * F - 2 * D)) + 0.011 * math.sin(r(M - 4 * D)))
    lat += (-0.173 * math.sin(r(F - 2 * D)) - 0.055 * math.sin(r(M - F - 2 * D))
            - 0.046 * math.sin(r(M + F - 2 * D)) + 0.033 * math.sin(r(F + 2 * D))
            + 0.017 * math.sin(r(2 * M + F)))
    rad_dist += -0.58 * math.cos(r(M - 2 * D)) - 0.46 * math.cos(r(2 * D))

    xg = rad_dist * math.cos(r(lon)) * math.cos(r(lat))
    yg = rad_dist * math.sin(r(lon)) * math.cos(r(lat))
    zg = rad_dist * math.sin(r(lat))
    return np.array([xg, yg, zg]) * EARTH_RADIUS_AU


def geocenter_position(jd: float) -> np.ndarray:
    """Heliocentric ecliptic position of Earth's *centre* (AU).

    Standish's "Earth" is the Earth-Moon **barycentre**; the geocentre is offset
    toward the Moon by the Moon's mass fraction of the Earth-Moon vector. This
    matters for topocentric orbit determination, where the ~4,700 km barycentre
    offset would otherwise dominate the residuals for nearby objects.
    """
    return earth_position(jd) - _MOON_MASS_FRACTION * moon_position_geocentric(jd)
