"""Build mixed biomedical:general training mini-batches.

Each batch holds `batch_queries` queries with `k_docs` candidates each. Biomedical rows carry
the grader's 1-10 grades (stratified pick over grade levels). General rows carry the frozen
encoder's cosine between the stored query and document vectors (pos/neg treated as a two-level
grade for the same stratified pick). The general count per batch cycles so the aggregate
biomedical:general ratio equals `--ratio` while every batch keeps the same size.

Inputs:
  <biomed-root>/_converted/<lane>/{queries,scores,doc_meta_cache}.jsonl
  <general-sample-dir>/<slice>.jsonl                 query, pos, neg texts per row
Outputs: <out> (batches.jsonl), <out>.summary.json, <out>.leftover.jsonl"""
import argparse, glob, json, os, random
from collections import defaultdict

import numpy as np


def _stratified_pick(items, k, rng):
    """Pick k of (id, grade) by round-robin over grade levels, highest grade first.

    Each lap takes one item from every grade level that still has items, so scarce high
    grades are kept and the remainder fills from the abundant low grades.
    """
    if k >= len(items):
        return list(items)
    by_grade = {}
    for w, g in items:
        by_grade.setdefault(g, []).append((w, g))
    for g in by_grade:
        rng.shuffle(by_grade[g])
    levels = sorted(by_grade, reverse=True)
    out = []
    while len(out) < k:
        progressed = False
        for g in levels:
            if by_grade[g] and len(out) < k:
                out.append(by_grade[g].pop()); progressed = True
        if not progressed:
            break
    return out[:k]


def load_biomed(root, k_docs, seed, pick, lanes=None, split=None):
    """Load the graded biomedical pools from `<root>/_converted/<lane>/`, optionally filtered by lane
    name and by the per-query `split` field."""
    pools = []
    lane_dirs = sorted(glob.glob(os.path.join(root, "_converted", "*")))
    lane_dirs = [d for d in lane_dirs if os.path.isdir(d)]
    if lanes:
        want = set(lanes)
        found = {os.path.basename(d) for d in lane_dirs}
        missing = want - found
        if missing:
            raise SystemExit(f"FATAL: --lanes named {sorted(missing)} but {root}/_converted/ "
                             f"holds {sorted(found)}")
        skipped = sorted(found - want)
        lane_dirs = [d for d in lane_dirs if os.path.basename(d) in want]
        print(f"lane filter: using {sorted(want)}; SKIPPING {skipped}")
    else:
        print(f"lane filter: OFF -- using every lane in {root}/_converted/: "
              f"{[os.path.basename(d) for d in lane_dirs]}")
    for lane_dir in lane_dirs:
        lane = os.path.basename(lane_dir)
        qtext = {}
        n_split_skipped = 0
        for line in open(os.path.join(lane_dir, "queries.jsonl")):
            d = json.loads(line)
            if split is not None and "split" in d and d["split"] != split:
                n_split_skipped += 1
                continue
            qtext[d.get("id") or d.get("qid")] = d.get("text") or d.get("query")
        if split is not None:
            print(f"  {lane}: split={split!r} kept {len(qtext)}, skipped {n_split_skipped}"
                  + ("  (lane carries no `split` field -- filter inactive here)"
                     if n_split_skipped == 0 and len(qtext) else ""))
        meta = {}
        for line in open(os.path.join(lane_dir, "doc_meta_cache.jsonl")):
            d = json.loads(line)
            w = d.get("work_id") or d.get("id")
            meta[w] = ((d.get("title") or "") + " " + (d.get("abstract") or "")).strip()
        for line in open(os.path.join(lane_dir, "scores.jsonl")):
            s = json.loads(line)
            qid = s["id"]
            if qid not in qtext: continue
            items = [(w, float(g)) for w, g in s["scores"].items() if w in meta and meta[w]]
            if len(items) < max(4, k_docs // 2): continue
            rng = random.Random(f"{seed}:{qid}")
            sel = _stratified_pick(items, k_docs, rng) if pick == "stratified" else \
                sorted(items, key=lambda x: -x[1])[:k_docs]
            gr = [g for _, g in sel]
            live = sum(1 for i in range(len(gr)) for j in range(i + 1, len(gr)) if gr[i] != gr[j])
            pools.append({"lane": "biomed", "sub": lane, "qid": qid, "query": qtext[qid],
                          "docs": [meta[w] for w, _ in sel],
                          "teacher": gr, "teacher_kind": "luna_1_10",
                          "n_docs": len(gr), "n_live_pairs": live})
    return pools


def load_general(sample_dir, vec_root, k_docs, seed, pick):
    pools = []
    for idsf in sorted(glob.glob(os.path.join(vec_root, "*", "ids.json"))):
        p = os.path.dirname(idsf); sl = os.path.basename(p)
        meta = json.load(open(idsf))
        Df = np.load(f"{p}/frozen__docs.fp16.npy").astype(np.float32)
        Qf = np.load(f"{p}/frozen__queries.fp16.npy").astype(np.float32)
        rows = [json.loads(l) for l in open(os.path.join(sample_dir, f"{sl}.jsonl"))]
        byq = defaultdict(list)
        for i, m in enumerate(meta["doc_meta"]):
            byq[m["q"]].append((i, m["role"]))
        for qi, items in sorted(byq.items()):
            if qi >= len(Qf) or qi >= len(rows): continue
            r = rows[qi]
            texts = list(r["pos"]) + list(r["neg"])
            if len(texts) != len(items): continue
            gid = [(j, 1.0 if role == "pos" else 0.0) for j, (_, role) in enumerate(items)]
            rng = random.Random(f"{seed}:{sl}:{qi}")
            sel = _stratified_pick(gid, k_docs, rng) if pick == "stratified" else gid[:k_docs]
            rowsel = np.array([items[j][0] for j, _ in sel])
            teacher = (Df[rowsel] @ Qf[qi]).astype(float).tolist()
            pools.append({"lane": "general", "sub": sl, "qid": f"{sl}:{qi}",
                          "query": r["query"], "prompt": r.get("prompt", ""),
                          "docs": [texts[j] for j, _ in sel],
                          "teacher": teacher, "teacher_kind": "frozen_reasonembed_cos",
                          "is_pos": [bool(g) for _, g in sel],
                          "n_docs": len(sel)})
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biomed-root", required=True)
    ap.add_argument("--general-sample-dir", required=True)
    ap.add_argument("--general-vec-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=int, default=5, help="biomedical queries per 1 general")
    ap.add_argument("--batch-queries", type=int, default=32)
    ap.add_argument("--k-docs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pick", default="stratified", choices=("stratified", "head"))
    ap.add_argument("--general-pick", default="stratified", choices=("stratified", "head"))
    ap.add_argument("--lanes", default="",
                    help="comma-separated lane names under <root>/_converted/ to load; empty = every lane")
    ap.add_argument("--general-reuse", action="store_true",
                    help="size the epoch by the biomedical pools and cycle the general lane with reshuffled "
                         "repeat passes when it runs out; off = use each general pool at most once")
    ap.add_argument("--split", default="",
                    help="keep only query records whose `split` field equals this; empty = no filter")
    a = ap.parse_args()

    bio = load_biomed(a.biomed_root, a.k_docs, a.seed, a.pick,
                      lanes=[x for x in a.lanes.split(",") if x] or None,
                      split=a.split or None)
    gen = load_general(a.general_sample_dir, a.general_vec_root, a.k_docs, a.seed, a.general_pick)
    print(f"biomedical pools {len(bio)} | general pools {len(gen)}")

    rng = random.Random(a.seed)
    rng.shuffle(bio); rng.shuffle(gen)

    B = a.batch_queries

    base = B // (a.ratio + 1)
    rem = B - base * (a.ratio + 1)
    cycle = [base + 1] * rem + [base] * ((a.ratio + 1) - rem)
    if a.general_reuse:
        n_batches = int(len(bio) / (B - sum(cycle) / len(cycle)))
    else:
        n_batches = min(len(gen) // (sum(cycle) / len(cycle)), len(bio) / (B - sum(cycle) / len(cycle)))
        n_batches = int(n_batches)

    batches, bi, gi = [], 0, 0
    gen_order = list(gen); gen_passes = 1
    for b in range(n_batches):
        ng = cycle[b % len(cycle)]
        nb = B - ng
        if bi + nb > len(bio): break
        if gi + ng > len(gen_order):
            if not a.general_reuse: break
            rng.shuffle(gen_order); gi = 0; gen_passes += 1
        rows = bio[bi:bi + nb] + gen_order[gi:gi + ng]
        bi += nb; gi += ng
        rng.shuffle(rows)
        batches.append({"batch": b, "n_biomed": nb, "n_general": ng, "rows": rows})

    nb_tot = sum(x["n_biomed"] for x in batches)
    ng_tot = sum(x["n_general"] for x in batches)
    realised = nb_tot / max(ng_tot, 1)
    if a.general_reuse:
        print(f"general-reuse ON: {gen_passes} passes over {len(gen)} general pools")
    print(f"batches {len(batches)} x {B} | biomed {nb_tot} general {ng_tot} "
          f"| realised ratio {realised:.4f} (target {a.ratio})")
    assert abs(realised - a.ratio) < 0.02, f"ratio drifted to {realised}"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)


    left_bio = bio[bi:]
    left_gen = [] if a.general_reuse else gen[gi:]
    lp = a.out.replace(".jsonl", ".leftover.jsonl")
    with open(lp, "w") as fh:
        for r in left_bio + left_gen:
            fh.write(json.dumps(r) + "\n")
    print(f"leftover unused pools: biomed {len(left_bio)} general {len(left_gen)} -> {lp}")
    with open(a.out, "w") as fh:
        for x in batches:
            fh.write(json.dumps(x) + "\n")
    import statistics as _st
    _bd = [r["n_docs"] for x in batches for r in x["rows"] if r["lane"] == "biomed"]
    _bp = [r["n_live_pairs"] for x in batches for r in x["rows"] if r["lane"] == "biomed"]
    _gd = [r["n_docs"] for x in batches for r in x["rows"] if r["lane"] == "general"]
    print(f"biomed pool depth: median {int(_st.median(_bd))} min {min(_bd)} max {max(_bd)} | "
          f"live pairs median {int(_st.median(_bp))} min {min(_bp)} max {max(_bp)}")
    json.dump({"n_batches": len(batches), "batch_queries": B, "k_docs": a.k_docs,
               "biomed_depth": {"median": _st.median(_bd), "min": min(_bd), "max": max(_bd),
                                "mean": _st.mean(_bd)},
               "biomed_live_pairs": {"median": _st.median(_bp), "min": min(_bp), "max": max(_bp),
                                     "mean": _st.mean(_bp)},
               "general_depth": {"median": _st.median(_gd), "min": min(_gd), "max": max(_gd)},
               "general_reuse": bool(a.general_reuse), "general_passes": gen_passes,
               "leftover_biomed": len(left_bio), "leftover_general": len(left_gen),
               "ratio_target": a.ratio, "ratio_realised": realised,
               "n_biomed": nb_tot, "n_general": ng_tot, "seed": a.seed,
               "pick": a.pick, "general_pick": a.general_pick,


               "lanes": a.lanes or "ALL", "split": a.split or "ALL",
               "biomed_pools_available": len(bio), "general_pools_available": len(gen)},
              open(a.out.replace(".jsonl", ".summary.json"), "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
