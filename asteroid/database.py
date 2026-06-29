"""Local asteroid database: a SQLite cache seeded from a bundled JSON file.

A :class:`Body` is one stored object — its Keplerian elements plus identity and
physical data. The package ships ``data/seed.json`` (a curated set of famous
asteroids) which is loaded into a writable SQLite database on first use, so the
tool works fully offline out of the box. Looking up an object not in the cache
is the CLI's cue to fetch it live from JPL (see :mod:`asteroid.fetch`) and
upsert it here.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, asdict, field
from importlib import resources
from pathlib import Path
from typing import Optional

from .constants import RAD2DEG
from .propagate import OrbitalElements


# --------------------------------------------------------------------------- #
# The stored record
# --------------------------------------------------------------------------- #
@dataclass
class Body:
    """An asteroid/comet record: identity, orbital elements, physical data.

    Angles are in degrees and ``a``/distances in AU (JPL's native units); the
    propagator-ready form is produced by :meth:`to_elements`.
    """

    key: str                       # lowercase lookup key
    name: str                      # display name, e.g. "99942 Apophis"
    a: float                       # semi-major axis (AU)
    e: float                       # eccentricity
    inc_deg: float                 # inclination (deg)
    node_deg: float                # longitude of ascending node (deg)
    argp_deg: float                # argument of perihelion (deg)
    M0_deg: float                  # mean anomaly at epoch (deg)
    epoch: float                   # epoch (Julian Date)
    n_deg: Optional[float] = None  # mean motion (deg/day)
    per_days: Optional[float] = None
    q: Optional[float] = None      # perihelion distance (AU)
    ad: Optional[float] = None     # aphelion distance (AU)
    H: Optional[float] = None      # absolute magnitude
    albedo: Optional[float] = None
    diameter_km: Optional[float] = None
    neo: bool = False
    pha: bool = False
    orbit_class: str = ""
    moid: Optional[float] = None   # Earth MOID (AU)
    fullname: str = ""
    designation: str = ""
    spkid: str = ""
    source: str = ""               # "seed" or "JPL SBDB"
    fetched_at: str = ""

    def to_elements(self) -> OrbitalElements:
        return OrbitalElements.from_degrees(
            a=self.a, e=self.e, inc_deg=self.inc_deg, node_deg=self.node_deg,
            argp_deg=self.argp_deg, M0_deg=self.M0_deg, epoch=self.epoch,
            n_deg=self.n_deg, name=self.name,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Body":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_elements(cls, el: OrbitalElements, name: str,
                      source: str = "computed", **extra) -> "Body":
        """Build a stored record from propagator elements (e.g. a determined orbit)."""
        import math
        per = None if math.isinf(el.period_days) else el.period_days
        return cls(
            key=normalize_key(name), name=name,
            a=el.a, e=el.e, inc_deg=el.inc * RAD2DEG,
            node_deg=el.node * RAD2DEG, argp_deg=el.argp * RAD2DEG,
            M0_deg=el.M0 * RAD2DEG % 360.0, epoch=el.epoch,
            n_deg=el.mean_motion * RAD2DEG, per_days=per,
            q=el.perihelion, ad=(None if el.e >= 1.0 else el.aphelion),
            source=source, **extra,
        )


# --------------------------------------------------------------------------- #
# Database location & connection
# --------------------------------------------------------------------------- #
def db_home() -> Path:
    """Directory holding the writable database (overridable via ``ASTEROID_HOME``)."""
    home = os.environ.get("ASTEROID_HOME")
    base = Path(home) if home else Path.home() / ".asteroid"
    base.mkdir(parents=True, exist_ok=True)
    return base


def db_path() -> Path:
    override = os.environ.get("ASTEROID_DB")
    return Path(override) if override else db_home() / "asteroids.db"


_COLUMNS = list(Body.__dataclass_fields__.keys())  # type: ignore[attr-defined]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path()))
    conn.row_factory = sqlite3.Row
    return conn


_TEXT_COLUMNS = {"key", "name", "orbit_class", "fullname",
                 "designation", "spkid", "source", "fetched_at"}
_INT_COLUMNS = {"neo", "pha"}


def _column_type(col: str) -> str:
    if col in _TEXT_COLUMNS:
        return "TEXT"
    if col in _INT_COLUMNS:
        return "INTEGER"
    return "REAL"


def _create_schema(conn: sqlite3.Connection) -> None:
    defs = []
    for c in _COLUMNS:
        decl = f"{c} {_column_type(c)}"
        if c == "key":
            decl += " PRIMARY KEY"
        defs.append(decl)
    conn.execute(f"CREATE TABLE IF NOT EXISTS bodies ({', '.join(defs)})")
    conn.commit()


def init_db(seed: bool = True) -> None:
    """Create the schema and, if empty, load the bundled seed."""
    conn = _connect()
    try:
        _create_schema(conn)
        if seed:
            count = conn.execute("SELECT COUNT(*) FROM bodies").fetchone()[0]
            if count == 0:
                _load_seed_into(conn)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Seed handling
# --------------------------------------------------------------------------- #
def _seed_records() -> list:
    """Read the packaged seed JSON, returning a list of :class:`Body`."""
    try:
        text = resources.files("asteroid").joinpath("data/seed.json").read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        return []
    data = json.loads(text)
    return [Body.from_dict(d) for d in data.get("bodies", [])]


def _load_seed_into(conn: sqlite3.Connection) -> int:
    bodies = _seed_records()
    for b in bodies:
        _upsert(conn, b)
    conn.commit()
    return len(bodies)


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
def _row_values(b: Body) -> list:
    d = b.to_dict()
    out = []
    for c in _COLUMNS:
        v = d[c]
        if c in ("neo", "pha"):
            v = 1 if v else 0
        out.append(v)
    return out


def _upsert(conn: sqlite3.Connection, b: Body) -> None:
    placeholders = ", ".join("?" for _ in _COLUMNS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in _COLUMNS if c != "key")
    conn.execute(
        f"INSERT INTO bodies ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(key) DO UPDATE SET {updates}",
        _row_values(b),
    )


def upsert_body(b: Body) -> None:
    init_db()
    conn = _connect()
    try:
        _upsert(conn, b)
        conn.commit()
    finally:
        conn.close()


def _row_to_body(row: sqlite3.Row) -> Body:
    d = dict(row)
    d["neo"] = bool(d.get("neo"))
    d["pha"] = bool(d.get("pha"))
    return Body.from_dict(d)


def normalize_key(name: str) -> str:
    return " ".join(name.strip().lower().split())


# Friendly shorthands that don't appear verbatim in any stored field.
ALIASES = {
    "1i": "oumuamua",
    "2i": "borisov",
    "halley": "halley",
}


def get_body(name: str) -> Optional[Body]:
    """Look up a body by name/designation (case-insensitive). ``None`` if absent."""
    init_db()
    key = normalize_key(name)
    key = ALIASES.get(key, key)
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM bodies WHERE key = ?", (key,)).fetchone()
        if row is None:
            # Fall back to a fuzzy contains-match on name/designation.
            row = conn.execute(
                "SELECT * FROM bodies WHERE key LIKE ? OR "
                "lower(name) LIKE ? OR lower(designation) LIKE ? LIMIT 1",
                (f"%{key}%", f"%{key}%", f"%{key}%"),
            ).fetchone()
        return _row_to_body(row) if row else None
    finally:
        conn.close()


def list_bodies() -> list:
    """All stored bodies, ordered by perihelion distance (inner first)."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM bodies ORDER BY a ASC").fetchall()
        return [_row_to_body(r) for r in rows]
    finally:
        conn.close()


def reset_db() -> None:
    """Delete the database file (next use re-seeds it)."""
    p = db_path()
    if p.exists():
        p.unlink()
