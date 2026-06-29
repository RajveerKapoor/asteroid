"""Physical and astronomical constants used throughout the package.

All orbital mechanics here works in heliocentric units of astronomical units (AU)
and days, with the Gaussian gravitational constant defining the Sun's gravity.
This keeps Kepler's third law and JPL's published mean motions consistent.
"""

import math

# Gaussian gravitational constant (defining constant, AU^(3/2) / day).
# k^2 == GM_sun in units of AU^3 / day^2.  This is the IAU 1976 value still used
# for two-body Keplerian work and matches JPL's published mean motions.
GAUSS_K = 0.01720209895

# Standard gravitational parameter of the Sun in AU^3 / day^2.
MU_SUN = GAUSS_K * GAUSS_K  # ~2.9591220828559e-04

# Length of an astronomical unit in kilometres (IAU 2012 definition).
AU_KM = 149_597_870.7

# Seconds in a day.
DAY_S = 86_400.0

# AU per day -> km per second conversion factor.
AU_PER_DAY_TO_KM_S = AU_KM / DAY_S

# Speed of light, used for light-time correction in orbit determination.
SPEED_OF_LIGHT_KM_S = 299_792.458
SPEED_OF_LIGHT_AU_D = SPEED_OF_LIGHT_KM_S * DAY_S / AU_KM  # ~173.144633 AU/day

# Earth equatorial radius (km) and in AU, used to place ground observatories for
# topocentric corrections in orbit determination.
EARTH_RADIUS_KM = 6378.137
EARTH_RADIUS_AU = EARTH_RADIUS_KM / AU_KM  # ~4.26352e-5 AU

# Mean obliquity of the ecliptic at J2000.0 (degrees), used to rotate between
# the heliocentric ecliptic frame (where JPL elements live) and the equatorial
# frame (where RA/Dec live).
OBLIQUITY_J2000_DEG = 23.43928
OBLIQUITY_J2000_RAD = math.radians(OBLIQUITY_J2000_DEG)

# Julian Date of the J2000.0 epoch (2000-01-01 12:00 TT).
J2000_JD = 2451545.0

# Days per Julian century.
DAYS_PER_CENTURY = 36525.0

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
TWO_PI = 2.0 * math.pi
