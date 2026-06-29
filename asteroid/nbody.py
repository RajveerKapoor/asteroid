"""N-body numerical propagation: the Sun plus the eight major planets.

Two-body propagation (:mod:`asteroid.propagate`) advances osculating Keplerian
elements analytically. It is *exact* at the orbit's epoch, but it ignores the
gravity of the planets. Over a month that drift is a few hundred km; over a
couple of years it grows to tens of thousands of km; across a deep planetary
flyby (Apophis in April 2029) it diverges completely, because the close pass
bends the real trajectory in a way a fixed ellipse cannot represent.

This module instead integrates the equations of motion **numerically**, treating
the Sun and all eight planets as gravitating bodies, with an adaptive
Runge-Kutta (Dormand-Prince 5(4)) step that automatically shrinks near a close
approach and lengthens in quiet stretches.

Force model (heliocentric, AU and days):

    a = -mu_sun * r/|r|^3
        + sum_p  mu_p * ( (r_p - r)/|r_p - r|^3  -  r_p/|r_p|^3 )

The bracket is the standard third-body perturbation: the first term is the
planet's direct pull on the asteroid, the second (indirect) term is the Sun's
acceleration toward the planet, which appears because the heliocentric frame is
itself accelerating. Planet positions ``r_p`` come from the Standish approximate
ephemeris in :mod:`asteroid.bodies`.

**Accuracy.** Away from a close encounter this tracks NASA Horizons to roughly
tens of km, versus tens of thousands of km for two-body at the same epoch offset.
*Through* a deep flyby the result is limited not by the integrator but by the
arc-minute planet positions of the approximate ephemeris: a flyby amplifies that
ephemeris error into millions of km downstream, and no step size removes it.
Matching NASA across such an encounter needs DE440-grade planet positions, which
is out of scope here -- JPL's authoritative orbit is always one default
``asteroid <name>`` command away.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

from .bodies import PLANET_ORDER, planet_position
from .constants import MU_SUN
from .propagate import OrbitalElements, state_vector

# Sun / (planet-system) mass ratios (IAU / DE-series values). The Standish
# ephemeris in asteroid.bodies returns each planet's *system* barycentre -- in
# particular "Earth" is the Earth-Moon barycentre -- so the system mass is the
# physically consistent perturber mass.
_MASS_RATIO = {
    "Mercury": 6_023_600.0,
    "Venus": 408_523.71,
    "Earth": 328_900.56,        # Earth-Moon barycentre
    "Mars": 3_098_708.0,
    "Jupiter": 1_047.3486,
    "Saturn": 3_497.898,
    "Uranus": 22_902.98,
    "Neptune": 19_412.24,
}

# Gravitational parameter mu = GM of each planet in AU^3 / day^2.
PLANET_GM = {name: MU_SUN / ratio for name, ratio in _MASS_RATIO.items()}


def acceleration(r: np.ndarray, jd: float,
                 perturbers: Iterable[str] = PLANET_ORDER) -> np.ndarray:
    """Heliocentric acceleration (AU/day^2) at position ``r`` and time ``jd``.

    Sums the Sun's pull and each planet's direct + indirect perturbation. The
    planets are taken from the approximate ephemeris at ``jd``.
    """
    r = np.asarray(r, dtype=float)
    acc = (-MU_SUN / (r @ r) ** 1.5) * r
    for name in perturbers:
        rp = planet_position(name, jd)
        d = rp - r
        acc += PLANET_GM[name] * (d / (d @ d) ** 1.5 - rp / (rp @ rp) ** 1.5)
    return acc


def _deriv(state: np.ndarray, jd: float,
           perturbers: Iterable[str]) -> np.ndarray:
    """Time derivative of the 6-vector state ``[r, v]`` -> ``[v, a]``."""
    out = np.empty(6)
    out[:3] = state[3:]
    out[3:] = acceleration(state[:3], jd, perturbers)
    return out


# --------------------------------------------------------------------------- #
# Dormand-Prince 5(4) tableau (the coefficients behind MATLAB's ode45 and
# SciPy's RK45). Seven stages with the First-Same-As-Last property, so each
# accepted step costs six new derivative evaluations.
# --------------------------------------------------------------------------- #
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
# 5th-order solution weights (== last stage row, the FSAL property).
_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
# 4th-order weights, for the embedded error estimate.
_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640,
       -92097 / 339200, 187 / 2100, 1 / 40)
# Difference b5 - b4, applied directly to the stages to get the error vector.
_E = tuple(b5 - b4 for b5, b4 in zip(_B5, _B4))


def _dopri_step(state: np.ndarray, jd: float, h: float,
                k1: np.ndarray, perturbers: Iterable[str]):
    """One Dormand-Prince step of size ``h``.

    ``k1`` is the derivative at the start of the step (reused from the previous
    step via FSAL). Returns ``(state_next, err_vector, k_last)`` where ``k_last``
    is the derivative at the step end (the next step's ``k1``).
    """
    k = [k1]
    for i in range(1, 7):
        yi = state + h * sum(_A[i][j] * k[j] for j in range(i))
        k.append(_deriv(yi, jd + _C[i] * h, perturbers))
    state_next = state + h * sum(_B5[i] * k[i] for i in range(7))
    err = h * sum(_E[i] * k[i] for i in range(7))
    return state_next, err, k[6]


def integrate(state0: np.ndarray, jd0: float, jd1: float,
              rtol: float = 1e-10, atol: float = 1e-12,
              max_step: Optional[float] = None,
              perturbers: Iterable[str] = PLANET_ORDER) -> np.ndarray:
    """Integrate the state ``[r, v]`` from ``jd0`` to ``jd1`` and return it.

    Adaptive step control keeps the local error per step near ``rtol``/``atol``.
    Works in either time direction. ``max_step`` (days) optionally caps the step
    so a long quiet stretch cannot overshoot a feature of interest.
    """
    state = np.asarray(state0, dtype=float).copy()
    if jd1 == jd0:
        return state
    perturbers = tuple(perturbers)

    direction = math.copysign(1.0, jd1 - jd0)
    total = abs(jd1 - jd0)
    cap = total if max_step is None else min(max_step, total)

    # Initial step guess: a small fraction of the span, capped.
    h = direction * min(cap, max(total / 100.0, 1e-3))
    k1 = _deriv(state, jd0, perturbers)
    jd = jd0
    safety, min_factor, max_factor = 0.9, 0.2, 5.0

    while (jd1 - jd) * direction > 0:
        if abs(h) > (jd1 - jd) * direction:      # don't overshoot the target
            h = jd1 - jd
        state_next, err, k_last = _dopri_step(state, jd, h, k1, perturbers)

        scale = atol + rtol * np.maximum(np.abs(state), np.abs(state_next))
        err_norm = math.sqrt(np.mean((err / scale) ** 2))

        if err_norm <= 1.0:                       # accept
            jd += h
            state = state_next
            k1 = k_last                           # FSAL: reuse end derivative
            if err_norm == 0.0:
                factor = max_factor
            else:
                factor = min(max_factor, safety * err_norm ** -0.2)
            h *= factor
        else:                                     # reject, shrink, retry
            h *= max(min_factor, safety * err_norm ** -0.2)

        if max_step is not None and abs(h) > max_step:
            h = math.copysign(max_step, h)
        if abs(h) < 1e-9:                          # guard against stalling
            raise RuntimeError("n-body step underflow; integration stalled")

    return state


def propagate_state(r0: np.ndarray, v0: np.ndarray, jd0: float, jd1: float,
                    **kw) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a heliocentric state from ``jd0`` to ``jd1``; returns ``(r, v)``."""
    state = integrate(np.concatenate([r0, v0]), jd0, jd1, **kw)
    return state[:3], state[3:]


def propagate_elements(elements: OrbitalElements, jd: float,
                       **kw) -> tuple[np.ndarray, np.ndarray]:
    """N-body state ``(position_AU, velocity_AU_per_day)`` of ``elements`` at ``jd``.

    The osculating elements give an exact state at their own epoch; that state is
    integrated under the full Sun + planets force model to ``jd``.
    """
    r0, v0 = state_vector(elements, elements.epoch)
    if jd == elements.epoch:
        return r0, v0
    return propagate_state(r0, v0, elements.epoch, jd, **kw)


def ephemeris(elements: OrbitalElements, jds: Iterable[float],
              **kw) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """N-body states at many times, integrating once through the sorted grid.

    Each requested time continues the integration from the previous one (rather
    than restarting from the epoch), so the whole ephemeris costs a single sweep
    forward and, if needed, a single sweep backward from the epoch.
    """
    epoch = elements.epoch
    r0, v0 = state_vector(elements, epoch)
    state0 = np.concatenate([r0, v0])

    targets = sorted(set(jds))
    forward = [t for t in targets if t >= epoch]
    backward = [t for t in reversed([t for t in targets if t < epoch])]

    results: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    state, jd = state0.copy(), epoch
    for t in forward:
        state = integrate(state, jd, t, **kw)
        jd = t
        results[t] = (state[:3].copy(), state[3:].copy())

    state, jd = state0.copy(), epoch
    for t in backward:
        state = integrate(state, jd, t, **kw)
        jd = t
        results[t] = (state[:3].copy(), state[3:].copy())

    return [(t, results[t][0], results[t][1]) for t in targets]
