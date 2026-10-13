from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rag_engine.adapters.embeddings.hashing import HashingEmbedder
from rag_engine.evaluation import cli
from rag_engine.evaluation.datasets import Dataset, load_beir, sha256_of
from rag_engine.evaluation.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from rag_engine.evaluation.runner import run

QRELS = {"a": 2, "b": 1, "z": 0}


def test_recall() -> None:
    assert recall_at_k(["a", "x", "b"], QRELS, 2) == 0.5
    assert recall_at_k(["a", "x", "b"], QRELS, 3) == 1.0
    assert recall_at_k(["a"], {"a": 0}, 1) == 0.0


def test_mrr() -> None:
    assert mrr_at_k(["x", "z", "b"], QRELS, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y"], QRELS, 10) == 0.0


def test_ndcg_uses_graded_gains() -> None:
    assert ndcg_at_k(["a", "b"], QRELS, 10) == pytest.approx(1.0)
    swapped = (1 / math.log2(2) + 2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], QRELS, 10) == pytest.approx(swapped)
    assert ndcg_at_k(["x"], {}, 10) == 0.0


def test_hit() -> None:
    assert hit_at_k(["x", "b"], QRELS, 2) == 1.0
    assert hit_at_k(["x", "z"], QRELS, 2) == 0.0


def test_nimbus_dataset_is_consistent(nimbus: Dataset) -> None:
    doc_ids = {d.id for d in nimbus.documents}
    assert len(nimbus.documents) == 31
    assert len(nimbus.queries) == 40
    assert set(nimbus.queries) == set(nimbus.qrels)
    assert all(set(judged) <= doc_ids for judged in nimbus.qrels.values())


def test_runner_on_nimbus_is_reproducible(nimbus: Dataset) -> None:
    first, stats = run(nimbus, {"hashing": HashingEmbedder(512)})
    second, _ = run(nimbus, {"hashing": HashingEmbedder(512)})
    assert [r.config for r in first] == ["BM25", "Dense: hashing", "Hybrid RRF: BM25 + hashing"]
    assert [r.ndcg_at_10 for r in first] == [r.ndcg_at_10 for r in second]
    assert stats[0].documents == 31
    bm25 = first[0]
    # Regression guard: the bundled set must stay solvable by lexical search.
    assert bm25.ndcg_at_10 > 0.7
    assert bm25.queries == 40


def test_cli_writes_json_and_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "results.json"
    cli.main(["--dataset", "nimbus", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "| BM25 |" in printed
    assert "Indexing with hashing-512" in printed
    payload = json.loads(out.read_text())
    assert payload["dataset"] == "nimbus"
    assert len(payload["results"]) == 3


def test_cli_accepts_a_beir_folder_and_rejects_unknown_embedders(tmp_path: Path) -> None:
    folder = tmp_path / "mini"
    (folder / "qrels").mkdir(parents=True)
    (folder / "corpus.jsonl").write_text('{"_id": "d1", "text": "hello world"}\n')
    (folder / "queries.jsonl").write_text(
        '{"_id": "q1", "text": "hello"}\n{"_id": "q2", "text": "x"}\n'
    )
    (folder / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    dataset = cli.load_dataset(str(folder), tmp_path)
    assert dataset.queries == {"q1": "hello"}  # unjudged queries are skipped
    assert load_beir(folder).name == "mini"
    with pytest.raises(SystemExit):
        cli.build_embedders(["word2vec"])


def test_sha256(tmp_path: Path) -> None:
    file = tmp_path / "f"
    file.write_bytes(b"abc")
    assert sha256_of(file).startswith("ba7816bf")
