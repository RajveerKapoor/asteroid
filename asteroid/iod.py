"""Orbit determination: compute an orbit from raw sky observations.

This is the *inverse* of the rest of the package. Instead of propagating known
elements, it takes a handful of plain observations — each a time plus a right
ascension and declination — and works out the orbit, exactly the problem Gauss
solved in 1801 to recover the newly-lost Ceres.

Two stages:

1. **Gauss's angles-only method** (:func:`gauss_preliminary`) turns three
   observations into a preliminary heliocentric state vector. It solves an
   eighth-degree polynomial for the body's distance at the middle observation.
2. **Differential correction** (:func:`differential_correction`) then refines
   that state by least-squares so it best fits *all* the observations, reporting
   the residual in arc-seconds.

Observations are treated as geocentric (observer at Earth's centre) by default;
each :class:`Observation` may carry its own heliocentric observer position.
Light-time is corrected for in the refinement stage.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import frames
from .bodies import earth_position, geocenter_position
from .constants import (MU_SUN, RAD2DEG, DEG2RAD, AU_KM, SPEED_OF_LIGHT_AU_D,
                        EARTH_RADIUS_AU)
from .propagate import OrbitalElements, state_vector, elements_from_state


@dataclass
class Observation:
    """One astrometric observation: a time and a sky direction.

    ``ra_deg``/``dec_deg`` are geocentric/topocentric equatorial J2000 angles.
    ``observer`` is the observer's heliocentric *ecliptic* position (AU); if
    omitted it defaults to Earth's centre at ``jd``.
    """

    jd: float
    ra_deg: float
    dec_deg: float
    observer: Optional[np.ndarray] = None

    def observer_position(self) -> np.ndarray:
        return self.observer if self.observer is not None else earth_position(self.jd)

    def los_ecliptic(self) -> np.ndarray:
        """Unit line-of-sight vector in the ecliptic frame."""
        return radec_to_ecliptic_unit(self.ra_deg, self.dec_deg)


def radec_to_ecliptic_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    """(RA, Dec) -> unit vector, rotated from equatorial into the ecliptic frame."""
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    equ = np.array([math.cos(dec) * math.cos(ra),
                    math.cos(dec) * math.sin(ra),
                    math.sin(dec)])
    return frames.equatorial_to_ecliptic(equ)


def observer_position_topocentric(stn, jd: float, obscodes: dict) -> np.ndarray:
    """Heliocentric ecliptic position (AU) of a ground observatory at ``jd``.

    Combines Earth's geocentre with the observatory's offset, found from its
    parallax constants ``(longitude_east, rho*cos(phi'), rho*sin(phi'))`` and the
    local apparent sidereal time. Unknown or space-based codes (no parallax
    constants) fall back to the geocentre.
    """
    consts = obscodes.get(str(stn)) if stn is not None else None
    if not consts:
        return geocenter_position(jd)
    lon_east, rho_cos, rho_sin = consts
    if rho_cos == 0.0 and rho_sin == 0.0:        # code 500: the geocentre itself
        return geocenter_position(jd)
    lst = math.radians(frames.gmst_deg(jd) + lon_east)
    obs_equatorial = EARTH_RADIUS_AU * np.array(
        [rho_cos * math.cos(lst), rho_cos * math.sin(lst), rho_sin])
    return geocenter_position(jd) + frames.equatorial_to_ecliptic(obs_equatorial)


# --------------------------------------------------------------------------- #
# Stage 1: Gauss's method
# --------------------------------------------------------------------------- #
def gauss_preliminary(obs1: Observation, obs2: Observation, obs3: Observation,
                      mu: float = MU_SUN) -> tuple:
    """Preliminary heliocentric state ``(r2, v2)`` at the middle observation.

    Implements the classical Gauss angles-only method. Returns the ecliptic
    position (AU) and velocity (AU/day) at ``obs2.jd``.
    """
    t1, t2, t3 = obs1.jd, obs2.jd, obs3.jd
    tau1, tau3, tau = t1 - t2, t3 - t2, t3 - t1

    rho1, rho2, rho3 = (o.los_ecliptic() for o in (obs1, obs2, obs3))
    R1, R2, R3 = (o.observer_position() for o in (obs1, obs2, obs3))

    p1 = np.cross(rho2, rho3)
    p2 = np.cross(rho1, rho3)
    p3 = np.cross(rho1, rho2)
    D0 = float(np.dot(rho1, p1))
    if abs(D0) < 1e-14:
        raise ValueError("degenerate geometry (observations are coplanar/collinear)")

    D = np.array([[float(np.dot(Ri, pj)) for pj in (p1, p2, p3)]
                  for Ri in (R1, R2, R3)])
    D11, D12, D13 = D[0]
    D21, D22, D23 = D[1]
    D31, D32, D33 = D[2]

    A = (1.0 / D0) * (-D12 * tau3 / tau + D22 + D32 * tau1 / tau)
    B = (1.0 / (6.0 * D0)) * (D12 * (tau3**2 - tau**2) * tau3 / tau
                              + D32 * (tau**2 - tau1**2) * tau1 / tau)
    E = float(np.dot(R2, rho2))
    R2sq = float(np.dot(R2, R2))

    a_c = -(A * A + 2.0 * A * E + R2sq)
    b_c = -2.0 * mu * B * (A + E)
    c_c = -(mu * mu) * (B * B)

    # Octic: r2^8 + a_c r2^6 + b_c r2^3 + c_c = 0.
    coeffs = [1.0, 0.0, a_c, 0.0, 0.0, b_c, 0.0, 0.0, c_c]
    roots = np.roots(coeffs)
    candidates = sorted(float(r.real) for r in roots
                        if abs(r.imag) < 1e-9 and r.real > 0)
    if not candidates:
        raise ValueError("Gauss method found no positive real root for r2")

    best = None
    for r2mag in candidates:
        try:
            state = _state_from_r2(r2mag, A, B, D0, D, tau1, tau3, tau,
                                   rho1, rho2, rho3, R1, R2, R3, mu)
        except (ValueError, ZeroDivisionError):
            continue
        resid = _triplet_residual(state, t2, obs1, obs3)
        if best is None or resid < best[0]:
            best = (resid, state)
    if best is None:
        raise ValueError("Gauss method: no physical solution among roots")
    return best[1]


def _state_from_r2(r2mag, A, B, D0, D, tau1, tau3, tau,
                   rho1, rho2, rho3, R1, R2, R3, mu):
    """Ranges -> position/velocity for a given middle-distance ``r2mag``."""
    D11, D12, D13 = D[0]
    D21, D22, D23 = D[1]
    D31, D32, D33 = D[2]
    r2cubed = r2mag**3

    rho2_range = A + mu * B / r2cubed

    num1 = (6.0 * (D31 * tau1 / tau3 + D21 * tau / tau3) * r2cubed
            + mu * D31 * (tau**2 - tau1**2) * tau1 / tau3)
    den1 = 6.0 * r2cubed + mu * (tau**2 - tau3**2)
    rho1_range = (1.0 / D0) * (num1 / den1 - D11)

    num3 = (6.0 * (D13 * tau3 / tau1 - D23 * tau / tau1) * r2cubed
            + mu * D13 * (tau**2 - tau3**2) * tau3 / tau1)
    den3 = 6.0 * r2cubed + mu * (tau**2 - tau1**2)
    rho3_range = (1.0 / D0) * (num3 / den3 - D33)

    if rho2_range <= 0 or rho1_range <= 0 or rho3_range <= 0:
        raise ValueError("non-physical (negative) range")

    r1 = R1 + rho1_range * rho1
    r2 = R2 + rho2_range * rho2
    r3 = R3 + rho3_range * rho3

    # Lagrange (f, g) series for the velocity at the middle point.
    f1 = 1.0 - 0.5 * mu / r2cubed * tau1**2
    f3 = 1.0 - 0.5 * mu / r2cubed * tau3**2
    g1 = tau1 - (1.0 / 6.0) * mu / r2cubed * tau1**3
    g3 = tau3 - (1.0 / 6.0) * mu / r2cubed * tau3**3
    denom = f1 * g3 - f3 * g1
    v2 = (-f3 * r1 + f1 * r3) / denom
    return r2, v2


def _triplet_residual(state, t_ref, obs1, obs3) -> float:
    """Angular residual (rad) of a preliminary orbit against the outer two obs."""
    r2, v2 = state
    el = elements_from_state(r2, v2, t_ref)
    total = 0.0
    for obs in (obs1, obs3):
        r = state_vector(el, obs.jd)[0]
        los = r - obs.observer_position()
        los /= np.linalg.norm(los)
        cos_ang = float(np.dot(los, obs.los_ecliptic()))
        total += math.acos(max(-1.0, min(1.0, cos_ang)))
    return total


# --------------------------------------------------------------------------- #
# Stage 2: differential correction
# --------------------------------------------------------------------------- #
def predict_radec(r_ref: np.ndarray, v_ref: np.ndarray, t_ref: float,
                  obs: Observation, light_time: bool = True) -> tuple:
    """Predicted (RA, Dec) in degrees for a state, with optional light-time."""
    el = elements_from_state(r_ref, v_ref, t_ref)
    t_emit = obs.jd
    observer = obs.observer_position()
    r = state_vector(el, t_emit)[0]
    if light_time:
        for _ in range(2):
            delta = float(np.linalg.norm(r - observer))
            t_emit = obs.jd - delta / SPEED_OF_LIGHT_AU_D
            r = state_vector(el, t_emit)[0]
    geo = r - observer
    equ = frames.ecliptic_to_equatorial(geo)
    ra, dec = frames.equatorial_to_radec(equ)
    return ra, dec


def _residual_vector(r_ref, v_ref, t_ref, observations, light_time) -> np.ndarray:
    """Stacked (O - C) residuals in radians, RA weighted by cos(dec)."""
    res = []
    for obs in observations:
        ra_c, dec_c = predict_radec(r_ref, v_ref, t_ref, obs, light_time)
        dra = ((obs.ra_deg - ra_c + 180.0) % 360.0 - 180.0) * math.cos(
            math.radians(obs.dec_deg))
        ddec = obs.dec_deg - dec_c
        res.append(dra * DEG2RAD)
        res.append(ddec * DEG2RAD)
    return np.array(res)


def differential_correction(r0: np.ndarray, v0: np.ndarray, t_ref: float,
                            observations: List[Observation],
                            light_time: bool = True, max_iter: int = 20) -> tuple:
    """Least-squares refine a state to fit all observations.

    Returns ``(r, v, rms_arcsec, iterations)``. Uses a numerical Jacobian and
    Gauss-Newton steps with a simple backtracking line search for stability.
    """
    state = np.concatenate([np.asarray(r0, float), np.asarray(v0, float)])
    steps = np.array([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])  # AU and AU/day

    def residuals(s):
        return _residual_vector(s[:3], s[3:], t_ref, observations, light_time)

    def rms(res):
        return math.sqrt(float(np.mean(res**2))) * RAD2DEG * 3600.0

    res = residuals(state)
    best_rms = rms(res)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        # Numerical Jacobian of the residuals w.r.t. the 6 state components.
        J = np.empty((len(res), 6))
        for j in range(6):
            ds = np.zeros(6)
            ds[j] = steps[j]
            J[:, j] = (residuals(state + ds) - residuals(state - ds)) / (2 * steps[j])
        # Solve J * delta = -res  (residual = O - C, want it driven to zero).
        delta, *_ = np.linalg.lstsq(J, -res, rcond=None)

        # Backtracking line search.
        factor = 1.0
        for _ in range(8):
            trial = state + factor * delta
            trial_res = residuals(trial)
            trial_rms = rms(trial_res)
            if trial_rms < best_rms:
                state, res, best_rms = trial, trial_res, trial_rms
                break
            factor *= 0.5
        else:
            break  # no improvement -> converged/stuck

        if np.linalg.norm(delta) < 1e-11:
            break

    return state[:3], state[3:], best_rms, iterations


# --------------------------------------------------------------------------- #
# High-level driver
# --------------------------------------------------------------------------- #
@dataclass
class OrbitSolution:
    elements: OrbitalElements
    rms_arcsec: float
    n_obs: int
    arc_days: float
    iterations: int
    epoch: float


SHORT_ARC_DAYS = 90.0     # at/below this, fit all observations directly
WINDOW_DAYS = 45.0        # length of each candidate window for long arcs


def candidate_windows(observations: List[Observation], window_days: float = WINDOW_DAYS,
                      n_windows: int = 6, max_obs: int = 30) -> List[List[Observation]]:
    """Return the densest, time-spread observation windows from a long arc.

    Gauss needs closely-spaced observations and two-body propagation only holds
    over a limited span, so we don't fit years at once. We reduce to one
    observation per night, then pick the ``n_windows`` densest non-overlapping
    windows of ``window_days`` (each sampled to ``max_obs`` points) as candidates
    to determine and compare.
    """
    obs = sorted(observations, key=lambda o: o.jd)
    by_night = {}
    for o in obs:
        by_night.setdefault(int(round(o.jd)), o)
    nightly = sorted(by_night.values(), key=lambda o: o.jd)
    times = [o.jd for o in nightly]

    scored, j = [], 0
    for i in range(len(nightly)):
        if j < i:
            j = i
        while j < len(nightly) and times[j] < times[i] + window_days:
            j += 1
        scored.append((j - i, i))
    scored.sort(reverse=True)

    chosen, used_starts = [], []
    for count, i in scored:
        if count < 4:
            break
        if any(abs(times[i] - times[k]) < window_days for k in used_starts):
            continue
        used_starts.append(i)
        window = nightly[i:i + count]
        if len(window) > max_obs:
            idx = np.linspace(0, len(window) - 1, max_obs).astype(int)
            window = [window[k] for k in idx]
        chosen.append(window)
        if len(chosen) >= n_windows:
            break
    return chosen or [nightly[:max_obs]]


def _determine_on(obs: List[Observation], name: str, light_time: bool) -> OrbitSolution:
    """Run Gauss + differential correction on one set of observations."""
    obs = sorted(obs, key=lambda o: o.jd)
    triplet = (obs[0], obs[len(obs) // 2], obs[-1])
    r2, v2 = gauss_preliminary(*triplet)
    t_ref = triplet[1].jd
    r, v, rms, iters = differential_correction(r2, v2, t_ref, obs, light_time)
    el = elements_from_state(r, v, t_ref, name=name)
    return OrbitSolution(elements=el, rms_arcsec=rms, n_obs=len(obs),
                         arc_days=obs[-1].jd - obs[0].jd, iterations=iters, epoch=t_ref)


def determine_orbit(observations: List[Observation], name: str = "",
                    light_time: bool = True) -> OrbitSolution:
    """Determine an orbit from observations (Gauss + differential correction).

    Short arcs are fit directly. For long arcs (years of data), several dense
    candidate windows are tried and the best-fitting (lowest-residual) solution
    is returned — this avoids windows spanning a perturbed close approach where a
    two-body fit can't converge.
    """
    if len(observations) < 3:
        raise ValueError("need at least 3 observations to determine an orbit")
    obs = sorted(observations, key=lambda o: o.jd)

    if obs[-1].jd - obs[0].jd <= SHORT_ARC_DAYS:
        return _determine_on(obs, name, light_time)

    best = None
    for window in candidate_windows(obs):
        if len(window) < 3:
            continue
        try:
            sol = _determine_on(window, name, light_time)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue
        if best is None or sol.rms_arcsec < best.rms_arcsec:
            best = sol
    if best is None:
        raise ValueError("could not determine an orbit from these observations")
    return best


# --------------------------------------------------------------------------- #
# Observation file parsing
# --------------------------------------------------------------------------- #
def _parse_angle(token: str, is_ra: bool) -> float:
    """Parse an angle in sexagesimal (``HH:MM:SS`` / ``±DD:MM:SS``) or decimal.

    For RA, sexagesimal is interpreted in hours and converted to degrees.
    """
    token = token.strip()
    if ":" in token:
        sign = -1.0 if token.lstrip().startswith("-") else 1.0
        parts = [float(p) for p in token.replace("+", "").lstrip("-").split(":")]
        while len(parts) < 3:
            parts.append(0.0)
        deg = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
        deg *= sign
        return deg * 15.0 if is_ra else deg
    return float(token)


def parse_observation_file(path: str) -> List[Observation]:
    """Read observations from a text file.

    Each non-comment line has three comma- or whitespace-separated fields::

        <UTC date>, <RA>, <Dec>

    where the date is ``YYYY-MM-DD[ HH:MM[:SS]]`` (or a Julian Date), RA is
    sexagesimal hours (``HH:MM:SS``) or decimal degrees, and Dec is sexagesimal
    degrees (``±DD:MM:SS``) or decimal degrees. Lines starting with ``#`` are
    ignored.
    """
    observations: List[Observation] = []
    text = Path(path).read_text()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "," in line:
            fields = [f.strip() for f in line.split(",")]
        else:
            # Whitespace-separated: the date may itself contain a space.
            m = re.match(r"^(\S+(?:[ T]\S+)?)\s+(\S+)\s+(\S+)$", line)
            if not m:
                raise ValueError(f"cannot parse observation line: {raw!r}")
            fields = [m.group(1), m.group(2), m.group(3)]
        if len(fields) < 3:
            raise ValueError(f"observation needs date, RA, Dec: {raw!r}")
        jd = frames.parse_date(fields[0])
        ra = _parse_angle(fields[1], is_ra=True)
        dec = _parse_angle(fields[2], is_ra=False)
        observations.append(Observation(jd=jd, ra_deg=ra, dec_deg=dec))
    if not observations:
        raise ValueError(f"no observations found in {path}")
    return observations
