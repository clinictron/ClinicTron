"""08_assemble_union.py — join the converted lanes and build the training batches.

Driven by one YAML config:
    lanes:               name -> lane folder holding queries.jsonl + scores.jsonl + doc_meta_cache.jsonl
                         (stage 07 output)
    general_sample_dir:  general-domain texts, one <slice>.jsonl per slice
    general_vec_root:    frozen general-domain vectors, one <slice>/ folder per slice
    ratio, batch_queries, k_docs, seed, pick, general_pick, general_reuse, split
    out_dir:             where batches.jsonl + batches.summary.json are written
    expect:              optional {n_batches, n_biomed, n_general}; the run fails if the summary differs

Relative paths in the config resolve against the config file's folder.
Steps: (1) create <out_dir>/assembly/_converted/<lane> -> lane folder (symlinks, recorded in
assembly_manifest.json), (2) run build_mixed_batches.py with the config values,
(3) read batches.summary.json and check it against `expect`.
The lane folders are only read.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--python", default=sys.executable, help="interpreter with numpy installed")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    base = Path(a.config).resolve().parent
    rel = lambda p: str((base / os.path.expandvars(str(p))).resolve())

    out = Path(rel(cfg["out_dir"])); out.mkdir(parents=True, exist_ok=True)
    conv = out / "assembly" / "_converted"; conv.mkdir(parents=True, exist_ok=True)
    manifest = {"config": os.path.basename(a.config), "lanes": {}}
    for name, path in cfg["lanes"].items():
        src = Path(rel(path))
        for f in ("queries.jsonl", "scores.jsonl"):
            if not (src / f).exists():
                sys.exit(f"FATAL: lane {name}: missing {src / f}")
        dst = conv / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        n_q = sum(1 for _ in open(src / "queries.jsonl"))
        manifest["lanes"][name] = {"path": str(path), "n_queries": n_q}
        print(f"[assemble] lane {name}: {n_q} queries <- {path}")
    json.dump(manifest, open(out / "assembly_manifest.json", "w"), indent=1)

    cmd = [a.python, str(HERE / "build_mixed_batches.py"),
           "--biomed-root", str(out / "assembly"),
           "--general-sample-dir", rel(cfg["general_sample_dir"]),
           "--general-vec-root", rel(cfg["general_vec_root"]),
           "--out", str(out / "batches.jsonl"),
           "--lanes", ",".join(cfg["lanes"].keys()),
           "--split", cfg.get("split", "train"),
           "--ratio", str(cfg["ratio"]),
           "--batch-queries", str(cfg["batch_queries"]),
           "--k-docs", str(cfg["k_docs"]),
           "--seed", str(cfg["seed"]),
           "--pick", cfg.get("pick", "stratified"),
           "--general-pick", cfg.get("general_pick", "stratified")]
    if cfg.get("general_reuse", False):
        cmd.append("--general-reuse")
    print("[assemble] running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        sys.exit(f"FATAL: build_mixed_batches exited {r.returncode}")

    s = json.load(open(out / "batches.summary.json"))
    print(f"[assemble] batches={s['n_batches']} biomed={s['n_biomed']} general={s['n_general']} "
          f"ratio={s['ratio_realised']} seed={s['seed']} lanes={s['lanes']}")
    exp = cfg.get("expect") or {}
    bad = {k: (s.get(k), v) for k, v in exp.items() if s.get(k) != v}
    if bad:
        sys.exit(f"FATAL: summary differs from expect: {bad}")
    if exp:
        print("[assemble] expect check PASSED:", exp)


if __name__ == "__main__":
    main()
