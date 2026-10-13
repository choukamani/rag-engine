"""Load the bundled Nimbus help-center corpus into a running rag-engine API."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "nimbus" / "corpus.jsonl"


def main() -> None:
    base_url = os.getenv("RAG_URL", "http://localhost:8000")
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]
    payload = {
        "documents": [{"id": r["_id"], "title": r["title"], "text": r["text"]} for r in rows]
    }
    request = urllib.request.Request(  # noqa: S310 - URL from operator configuration
        f"{base_url}/documents",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        sys.stdout.write(response.read().decode() + "\n")


if __name__ == "__main__":
    main()
