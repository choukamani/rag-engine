"""Load retrieval benchmarks in the BEIR format (corpus.jsonl, queries.jsonl, qrels/<split>.tsv).

Two datasets are supported out of the box:

* `nimbus`: a small corpus written for this repository and versioned with it
  (offline, runs in CI in seconds).
* `scifact`: the public BEIR SciFact benchmark (5 183 abstracts, 300 test
  queries), downloaded on demand and verified by SHA-256 before use.
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rag_engine.domain import Document

SCIFACT_URL: Final = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
SCIFACT_SHA256: Final = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: list[Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    """query id -> {document id -> relevance grade}"""


def load_beir(path: Path, *, split: str = "test", name: str | None = None) -> Dataset:
    documents = []
    with (path / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            documents.append(Document(str(row["_id"]), row["text"], row.get("title", "")))

    qrels: defaultdict[str, dict[str, int]] = defaultdict(dict)
    with (path / "qrels" / f"{split}.tsv").open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for query_id, doc_id, score in reader:
            qrels[query_id][doc_id] = int(score)

    queries = {}
    with (path / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row["_id"]) in qrels:
                queries[str(row["_id"])] = row["text"]

    return Dataset(name or path.name, documents, queries, dict(qrels))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def download_scifact(cache_dir: Path) -> Path:  # pragma: no cover - network
    target = cache_dir / "scifact"
    if (target / "corpus.jsonl").exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "scifact.zip"
    urllib.request.urlretrieve(SCIFACT_URL, archive)
    actual = sha256_of(archive)
    if actual != SCIFACT_SHA256:
        archive.unlink()
        raise RuntimeError(f"scifact.zip checksum mismatch: {actual}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(cache_dir)
    return target
