#!/usr/bin/env python3
"""03_scan_base_encoder.py — rank the corpus for every query text with the frozen base encoder.

Each training row has three query texts: the written question and its two hypothetical
abstracts (sketches). Scan ids are `<qid>`, `<qid>#s1`, `<qid>#s2`. Every text is encoded
with the base encoder and ranked by exact cosine over the whole index (common/corpus.py).
Ids listed in the exclusion files (benchmark hold-out) are dropped after ranking and counted.

Writes rankings.jsonl ({"id", "slot", "qid", "fused": [[doc_id, score], ...]}), the query
texts (scan_input.jsonl), their vectors (scan_queries.npz) and scan_summary.json.
Rows with fewer than two sketches are refused unless --allow-missing-sketches.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import load_config                       # noqa: E402
from common.corpus import encode_queries, exact_search      # noqa: E402
from common.logging import Log, SkipLog                     # noqa: E402

SUFFIX = {"question": "", "sketch_1": "#s1", "sketch_2": "#s2"}


def scan_rows(row: dict) -> list[tuple[str, str, str]]:
    """(scan_id, slot, text) for the three query texts of one training row."""
    out = [(row["qid"], "question", row["text"])]
    for k, h in enumerate((row.get("sketches") or [])[:2], 1):
        text = f"{h.get('title', '')} {h.get('abstract', '')}".strip()
        if text:
            out.append((f"{row['qid']}{SUFFIX[f'sketch_{k}']}", f"sketch_{k}", text))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True, help="questions.jsonl from stage 02")
    ap.add_argument("--index-dir", help="published index directory (default: frozen_scan.index_dir)")
    ap.add_argument("--model-config", help="encoder model config (default: frozen_scan.model_config)")
    ap.add_argument("--exclude-ids", nargs="*", help="files of doc ids to drop, one per line "
                    "(default: frozen_scan.exclude_ids_files)")
    ap.add_argument("--device", help="cpu or cuda (default: frozen_scan.device, else cpu)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--allow-missing-sketches", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    fs = cfg["frozen_scan"]
    index_dir, model_cfg = a.index_dir or fs["index_dir"], a.model_config or fs["model_config"]
    if not index_dir:
        raise SystemExit("no base-encoder index: set GEN_BASE_INDEX or pass --index-dir")
    # a relative model config path is read from this generation directory
    model_cfg = str(Path(__file__).resolve().parent / model_cfg)
    device = a.device or fs.get("device", "cpu")
    depth = int(fs["depth"])
    out = Path(a.out_dir) if a.out_dir else cfg.out_dir("03_frozen_scan")
    out.mkdir(parents=True, exist_ok=True)
    log, skip = Log(out / "run.log"), SkipLog(out / "skips.log")

    rows = [json.loads(l) for l in open(a.questions) if l.strip()]
    short = [r["qid"] for r in rows if len(scan_rows(r)) < 3]
    for qid in short:
        skip.note("SKETCHES_MISSING", qid=qid, reason="needs the question AND two sketches")
    if short and not a.allow_missing_sketches:
        raise SystemExit(f"FATAL: {len(short)} rows lack two sketches, e.g. {short[:3]}; "
                         f"write the sketches first or pass --allow-missing-sketches")
    texts = [(r["qid"], *t) for r in rows for t in scan_rows(r)]
    with (out / "scan_input.jsonl").open("w") as fh:
        for qid, sid, slot, text in texts:
            fh.write(json.dumps({"_id": sid, "id": sid, "text": text, "qid": qid, "slot": slot},
                                ensure_ascii=False) + "\n")

    held = set()
    for p in (a.exclude_ids if a.exclude_ids is not None else fs.get("exclude_ids_files") or []):
        held |= {l.strip() for l in open(p) if l.strip()}

    t0 = time.time()
    Q = encode_queries(model_cfg, [t[3] for t in texts], fs["instruction"])
    t1 = time.time()
    np.savez(out / "scan_queries.npz", query_ids=np.array([t[1] for t in texts]), query_vectors=Q)
    ranked = exact_search(index_dir, Q, depth, device)
    t2 = time.time()

    dropped, by_slot = 0, collections.Counter()
    with (out / "rankings.jsonl").open("w") as fh:
        for (qid, sid, slot, _), hits in zip(texts, ranked):
            keep = [[d, s] for d, s in hits if d not in held]
            dropped += len(hits) - len(keep)
            by_slot[slot] += 1
            fh.write(json.dumps({"id": sid, "slot": slot, "qid": qid, "fused": keep}) + "\n")
    summary = {"rows": len(rows), "query_texts": len(texts), "by_slot": dict(by_slot),
               "rows_missing_sketches": len(short), "depth": depth, "device": device,
               "instruction": fs["instruction"], "excluded_ids": len(held),
               "excluded_hits_dropped": dropped, "encode_secs": round(t1 - t0, 1),
               "scan_secs": round(t2 - t1, 1), "skips": skip.summary()}
    log(f"SCAN {len(texts)} query texts, depth {depth}, dropped {dropped} held-out hits, "
        f"encode {t1 - t0:.0f}s scan {t2 - t1:.0f}s -> {out / 'rankings.jsonl'}")
    (out / "scan_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
