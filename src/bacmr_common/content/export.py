"""Write the JSON Schemas under `schemas/` from the models.

`uv run python -m bacmr_common.content.export` (a test fails when they drift)."""

from __future__ import annotations

import json
from pathlib import Path

from . import SCHEMA_FILES, build_json_schema


def main() -> None:
    out = Path(__file__).parent / "schemas"
    out.mkdir(exist_ok=True)
    for name, model in SCHEMA_FILES.items():
        text = json.dumps(build_json_schema(model), indent=2, ensure_ascii=False) + "\n"
        (out / name).write_text(text)
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
