"""Build asteroid/data/seed.json by fetching famous bodies from JPL SBDB.

Run once (needs network):  python scripts/build_seed.py
The resulting JSON is committed so the tool works fully offline.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asteroid.fetch import fetch_sbdb, FetchError  # noqa: E402

# Curated, famous, and physically diverse: main-belt giants, mission targets,
# planet-crossers, a Trojan, a centaur, a very eccentric comet, and two
# hyperbolic interstellar objects (to exercise the e>1 code path).
QUERIES = [
    "Ceres", "Pallas", "Juno", "Vesta", "Hygiea", "Psyche",
    "Eros", "Ida", "Gaspra", "Mathilde",
    "Bennu", "Ryugu", "Itokawa", "Apophis", "Didymos",
    "Phaethon", "Toutatis", "Geographos", "Icarus", "Florence",
    "624 Hektor", "Chiron",
    "1P",          # Halley (very eccentric ellipse)
    "1I",          # 'Oumuamua (hyperbolic, interstellar)
    "2I",          # Borisov (hyperbolic, interstellar)
]


def main() -> int:
    bodies = []
    for q in QUERIES:
        try:
            body = fetch_sbdb(q)
        except FetchError as exc:
            print(f"  SKIP {q!r}: {exc}")
            continue
        bodies.append(body.to_dict())
        flags = []
        if body.neo:
            flags.append("NEO")
        if body.pha:
            flags.append("PHA")
        if body.e >= 1.0:
            flags.append("hyperbolic")
        print(f"  OK   {body.name:30s} a={body.a:>9.4f} e={body.e:.4f} "
              f"{' '.join(flags)}")
        time.sleep(0.4)  # be polite to the API

    out = Path(__file__).resolve().parents[1] / "asteroid" / "data" / "seed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": "Seed database for asteroid-trajectory. Built from JPL SBDB. "
                    "Refresh with `asteroid --update` or scripts/build_seed.py.",
        "count": len(bodies),
        "bodies": bodies,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(bodies)} bodies to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
