"""asteroid-trajectory: compute asteroid trajectories from real orbital elements.

A small, dependency-light orbital-mechanics engine that loads Keplerian elements
(from a bundled database or live from NASA's JPL Small-Body Database) and
propagates them with a from-scratch two-body solver to produce positions,
ephemerides, sky coordinates, close approaches and visualizations.
"""

__version__ = "0.1.0"

from .constants import AU_KM, GAUSS_K, MU_SUN, OBLIQUITY_J2000_DEG  # noqa: F401

__all__ = ["__version__", "AU_KM", "GAUSS_K", "MU_SUN", "OBLIQUITY_J2000_DEG"]
