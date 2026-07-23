"""dump_snapshot.py — write the GET /worklist payload to the bundled UI snapshot.

`make snapshot` uses this so the deployed static sample
(src/ui/assets/worklist.sample.json) is reproducible from a clean state without
having to stand up a live server. It calls the SAME handler /worklist serves, so
the file is byte-for-byte the response the UI would get from the API (sorted by
break window), minus HTTP framing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.api.main import get_worklist  # noqa: E402  (the /worklist handler)

SNAPSHOT_PATH = ROOT / "src" / "ui" / "assets" / "worklist.sample.json"


def main() -> int:
    data = get_worklist()  # identical to GET /worklist (rows sorted by break window)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(data))
    funnel = data.get("escalation_funnel") or {}
    print(f"wrote {SNAPSHOT_PATH} ({SNAPSHOT_PATH.stat().st_size} bytes; "
          f"{len(data['worklist'])} rows, n_gated={funnel.get('n_gated')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
