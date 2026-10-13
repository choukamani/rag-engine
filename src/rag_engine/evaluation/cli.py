"""`rag-eval`: reproducible retrieval evaluation.

uv run rag-eval --dataset nimbus
uv run rag-eval --dataset scifact --embedders hashing,fastembed
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from rag_engine.adapters.embeddings.hashing import HashingEmbedder
from rag_engine.evaluation.datasets import Dataset, download_scifact, load_beir
from rag_engine.evaluation.runner import IndexStats, RunResult, run
from rag_engine.ports import Embedder

NIMBUS_PATH = Path(__file__).resolve().parents[3] / "evaluation" / "datasets" / "nimbus"


def load_dataset(name: str, cache: Path) -> Dataset:
    if name == "nimbus":
        return load_beir(NIMBUS_PATH, name="nimbus")
    if name == "scifact":  # pragma: no cover - network
        return load_beir(download_scifact(cache), name="scifact")
    return load_beir(Path(name))


def build_embedders(names: list[str]) -> dict[str, Embedder]:
    embedders: dict[str, Embedder] = {}
    for name in names:
        if name == "hashing":
            embedders["hashing-512"] = HashingEmbedder(512)
        elif name == "fastembed":  # pragma: no cover - model download
            from rag_engine.adapters.embeddings.fastembed import FastEmbedEmbedder

            embedders["bge-small-en-v1.5"] = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
        else:
            raise SystemExit(f"unknown embedder {name!r} (hashing, fastembed)")
    return embedders


def to_markdown(results: list[RunResult], stats: list[IndexStats]) -> str:
    lines = [
        "| Retrieval | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | Hit@5 | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r.config} | {r.ndcg_at_10:.3f} | {r.recall_at_10:.3f} | {r.recall_at_100:.3f} | "
        f"{r.mrr_at_10:.3f} | {r.hit_at_5:.3f} | {r.query_p50_ms} | {r.query_p95_ms} |"
        for r in results
    ]
    lines.append("")
    lines += [
        f"Indexing with {s.embedder}: {s.documents} documents, {s.chunks} chunks, {s.seconds} s"
        for s in stats
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nimbus", help="nimbus, scifact or a BEIR folder")
    parser.add_argument("--embedders", default="hashing", help="comma-separated: hashing,fastembed")
    parser.add_argument("--chunk-max-words", type=int, default=400)
    parser.add_argument("--cache", type=Path, default=Path(".cache/datasets"))
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset, args.cache)
    results, stats = run(
        dataset,
        build_embedders(args.embedders.split(",")),
        chunk_max_words=args.chunk_max_words,
    )
    print(f"Dataset: {dataset.name} ({len(dataset.documents)} documents, "
          f"{len(dataset.queries)} queries)\n")  # fmt: skip
    print(to_markdown(results, stats))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": dataset.name,
            "documents": len(dataset.documents),
            "queries": len(dataset.queries),
            "chunk_max_words": args.chunk_max_words,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "indexing": [s.__dict__ for s in stats],
            "results": [r.as_dict() for r in results],
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
