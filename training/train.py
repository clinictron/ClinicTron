"""LoRA fine-tune of a single-vector embedding model on mixed biomedical/general batches.

Every training batch holds two kinds of query rows:
  biomedical rows  each candidate document carries a graded relevance label (1-10); the loss is
                   the weighted pairwise rank loss `losses.pairwise_rank_loss`, which learns the
                   ORDER the grades imply, not their scale.
  general rows     each candidate carries the cosine score a frozen anchor model gave it; the loss
                   is KL(anchor || student) between temperature-scaled softmaxes over the
                   candidates. This term keeps the model close to the anchor's general-domain
                   ranking behaviour while it adapts to the biomedical rows.

  L = mean_over_biomedical(pairwise) + kl_weight * mean_over_general(KL)

Batches are pre-built offline (fixed biomedical:general ratio inside every batch, anchor scores
aligned to their documents) and replayed in file order.

Only LoRA adapters (plus, for NV-Embed-v2, its latent-attention pooling head) are trained.
A step holds batch_queries * K sequences, so gradient caching is required: pass 1 runs a no-grad
chunked forward and caches one embedding per sequence; the loss on those cached embeddings
backpropagates to d(loss)/d(embedding); pass 2 re-runs each chunk with grad and backpropagates the
cached embedding gradients into the parameters. The two forwards must be identical, so
lora_dropout must be 0.

Multi-GPU: the queries of each global batch are split across ranks. Each query is scored only
against its own candidates (no in-batch negatives), so each rank's loss is a sum of its own
per-query terms divided by the GLOBAL per-lane counts, and summing parameter gradients across
ranks reproduces the single-process gradient. The model is not wrapped in torch DDP because
gradient caching calls backward once per chunk; gradients are all-reduced explicitly instead.

Launch:  python train.py --config configs/X.json                         (one GPU)
         torchrun --nproc_per_node N train.py --config configs/X.json  (N GPUs)
"""
from __future__ import annotations

import argparse, dataclasses, json, os, random, time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from losses import pairwise_rank_loss


def query_prefix(task: str) -> str:
    """Query instruction prefix (E5 / NV-Embed template); the query text is appended to it."""
    return "Instruct: {task}\nQuery: ".format(task=task)


@dataclass
class PairwiseConfig:
    # No defaults: every field must be set by the config file (None = not set, refused in main).
    # configs/*.json hold the values used for the released models.
    # ---- data ----
    mixed_batches: str = None        # pre-built mixed batches, one JSON batch per line
    instruction: str = None          # query-side task instruction (documents carry none)

    # ---- loss ----
    sigma: float = None              # logistic scale of the pairwise loss; calibrated per model
                                     # as 2/median(|s_i-s_j|) over cross-grade pairs
    kl_weight: float = None          # weight on the general-lane KL term
    tau: float = None                # student softmax temperature for the KL term
    tau_teacher: float = None        # anchor softmax temperature; equal to tau
    gap_power: float = None
    gain: str = None
    position_alpha: float = None
    normalize: str = None

    # ---- model ----
    base_model: str = None
    base_kind: str = None            # "qwen3" | "nvembed_v2"
    pooling: str = None              # "last" (Qwen3 family) | "latent" (NV-Embed-v2)
    attention: str = None            # "causal" | "bidirectional"
    attn_implementation: str = None  # passed explicitly so numerics do not depend on installed kernels
    # Zero the instruction span in the query pooling mask. Must be False for last-token pooling,
    # which reads position sum(pool_mask)-1; masking would move that position into the query text.
    mask_instruction: bool = None
    expected_trainable_params: int = None # hard check on the trainable-parameter count; 0 = derive it
    lora_r: int = None
    lora_alpha: int = None
    lora_dropout: float = None       # must be 0 for gradient-cache determinism
    dtype: str = None
    gradient_checkpointing: bool = None
    max_len: int = None              # truncation threshold; batches pad only to their own longest

    # ---- optimization ----
    lr: float = None
    weight_decay: float = None
    batch_queries: int = None        # queries per optimizer step (global, across ranks)
    warmup_frac: float = None        # fraction of total steps
    grad_clip: float = None
    chunk_size: int = None           # max sequences per gradient-cache sub-forward
    chunk_tokens: int = None         # max padded tokens per sub-forward (chunk shrinks for long seqs)
    seed: int = None

    # ---- io ----
    out_dir: str = None
    log_every: int = None
    save_every: int = None
    max_steps: int = None            # total optimizer steps


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _doc_text(doc):
    """A document is stored as plain text or as an inverted index {word: [positions]}."""
    if isinstance(doc, str):
        return doc
    words = [None] * (1 + max(p for ps in doc.values() for p in ps))
    for w, ps in doc.items():
        for p in ps:
            words[p] = w
    return " ".join(words)


class MixedBatchDataset:
    """Replays the pre-built mixed batches verbatim.

    The biomedical:general ratio must hold within every batch (the loss takes a per-lane mean, so
    a batch without general rows would drop the KL term for that step), and each general row's
    anchor score vector must stay aligned with its documents. Batches are therefore built once,
    offline, and replayed in order.

      lane 0 = biomedical -> `grades` are relevance grades 1-10, pairwise rank loss
      lane 1 = general    -> `soft` are frozen anchor cosines, listwise KL

    `grades` is NaN on general rows and `soft` is 0.0 on biomedical rows; neither is read for
    that lane (see _mixed_lane_loss), and NaN makes a mis-wiring fail loudly.
    """

    def __init__(self, path):
        self.batches, self.rows = [], []
        for line in open(path):
            b = json.loads(line)
            idx = []
            for r in b["rows"]:
                r["docs"] = [_doc_text(d) for d in r["docs"]]
                idx.append(len(self.rows)); self.rows.append(r)
            self.batches.append(idx)
        sizes = {len(x) for x in self.batches}
        assert len(sizes) == 1, f"ragged batch sizes {sorted(sizes)}; the loop assumes one size"
        self.batch_queries = sizes.pop()
        nb = sum(1 for r in self.rows if r["lane"] == "biomed")
        ng = len(self.rows) - nb
        self.ratio = nb / max(ng, 1)
        print(f"[mixed] {len(self.batches)} batches x {self.batch_queries} | "
              f"biomed {nb} general {ng} | ratio {self.ratio:.3f}", flush=True)

    def __len__(self):
        return len(self.rows)

    def batch_indices(self, b):
        return self.batches[b]

    def lane_counts(self, b):
        """GLOBAL per-lane query counts (n_biomed, n_general) for batch `b`.

        Under multi-GPU a rank holds only a slice of the batch and often no general rows, so lane
        means are taken over the counts of the whole batch. Every rank reads the same file, so
        this is identical everywhere without communication.
        """
        ng = sum(1 for i in self.batches[b] if self.rows[i]["lane"] == "general")
        return len(self.batches[b]) - ng, ng

    def example(self, i, epoch=0):
        r = self.rows[i]
        n = len(r["docs"])
        general = r["lane"] == "general"
        return {"query": r["query"], "is_cot": False, "docs": r["docs"],
                "grades": [float("nan")] * n if general else [float(g) for g in r["teacher"]],
                "soft": [float(x) for x in r["teacher"]] if general else [0.0] * n,
                "lanes": 1 if general else 0,
                "strata": [0] * n}


def collate(batch, tok, max_len, instruction, mask_instruction: bool = True,
            pad_depth: bool = False):
    """Tokenize one batch. Right padding only (asserted).

    `mask_instruction=True` zeroes the instruction span in the query POOL mask (masked-mean /
    latent pooling). For last-token pooling it must be False.

    No EOS is appended here: text is tokenized with `add_special_tokens=True` and the model sees
    whatever the tokenizer adds (the Qwen3 tokenizer appends its own end-of-text token).

    Candidate lists may differ in length within a batch. `pad_depth=True` pads every list to the
    batch maximum: padded slots get grade NaN, an empty-string document, and False in
    `cand_mask`, so the loss can exclude them. Nothing is dropped and every anchor score stays
    aligned with its document.
    """
    if getattr(tok, "padding_side", "right") != "right":
        raise RuntimeError(
            f"tokenizer.padding_side={tok.padding_side!r}; right padding is required. "
            f"`_forward_chunks` trims chunks with ids[:, :L] (which cuts real tokens off a "
            f"left-padded batch) and last-token pooling indexes sum(mask)-1.")
    prefix = query_prefix(instruction)
    qs = [prefix + b["query"] for b in batch]
    docs, grades = [], []
    for b in batch:
        docs.extend(b["docs"])
        grades.append(b["grades"])
    if pad_depth:
        k = max(len(g) for g in grades)
        grades = torch.tensor(
            [list(g) + [float("nan")] * (k - len(g)) for g in grades], dtype=torch.float32)
        strata = torch.tensor(
            [list(b["strata"][:len(b["docs"])]) + [0] * (k - len(b["docs"])) for b in batch],
            dtype=torch.long)
        cand_mask = torch.zeros(len(batch), k, dtype=torch.bool)
        for i, b in enumerate(batch):
            cand_mask[i, :len(b["docs"])] = True
        trimmed = []
        for b in batch:
            trimmed.extend(b["docs"])
            trimmed.extend([""] * (k - len(b["docs"])))     # placeholder, masked out downstream
    else:
        k = min(len(g) for g in grades)
        grades = torch.tensor([g[:k] for g in grades], dtype=torch.float32)
        strata = torch.tensor([b["strata"][:k] for b in batch], dtype=torch.long)
        cand_mask = torch.ones(len(batch), k, dtype=torch.bool)
        trimmed = []
        for b in batch:
            trimmed.extend(b["docs"][:k])
    qenc = tok(qs, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
    denc = tok(trimmed, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
    # query pool mask; optionally excludes the instruction tokens (documents have none)
    qpool = qenc["attention_mask"].clone()
    if mask_instruction:
        pref_len = len(tok(prefix, add_special_tokens=False)["input_ids"])
        qpool[:, :pref_len] = 0
        qpool = torch.where(qpool.sum(1, keepdim=True) > 0, qpool, qenc["attention_mask"])
    out = {"q": qenc, "q_pool": qpool, "d": denc, "grades": grades, "strata": strata, "k": k,
           "cand_mask": cand_mask}
    for key in ("lanes", "soft"):
        if key in batch[0]:
            if key == "lanes":
                out[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)
            else:
                out[key] = torch.tensor(
                    [list(b[key]) + [0.0] * (k - len(b[key])) for b in batch], dtype=torch.float32)
    return out


# --------------------------------------------------------------------------- #
# step
# --------------------------------------------------------------------------- #
def _chunk_spans(lens, order, chunk, chunk_tokens):
    """Chunk boundaries over the length-sorted `order`.

    `chunk_tokens == 0` gives fixed-size chunks of `chunk`. With `chunk_tokens > 0` a chunk
    closes once `n_seqs * this_chunk's_longest` would exceed the budget, so chunk size adapts to
    sequence length and long documents do not run out of memory. Chunking does not change the
    objective: the loss is computed on the cached embeddings, and pass 1 and pass 2 derive their
    spans from the same `lens`, so both passes run identical forwards.
    """
    n = len(order)
    if not chunk_tokens or chunk_tokens <= 0:
        return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    spans, s = [], 0
    while s < n:
        longest = int(lens[order[s]].item())          # order is descending, so this is the max
        take = max(1, min(chunk, chunk_tokens // max(1, longest)))
        spans.append((s, min(s + take, n)))
        s += take
    return spans


def _forward_chunks(model, enc, pool, device, chunk, grad=False, chunk_tokens=0):
    """Chunked forward with length bucketing.

    Sequences are sorted by true (unpadded) length, chunked in that order, each chunk trimmed to
    its own longest member, and the embeddings scattered back to the original order. Output is
    order-identical to an unbucketed forward; only padding work is removed.
    """
    ids, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    pm = pool.to(device) if pool is not None else am
    n = ids.shape[0]
    ctx = torch.enable_grad() if grad else torch.no_grad()

    lens = am.sum(dim=1)                                  # true token count per sequence
    order = torch.argsort(lens, descending=True)          # long first: peak memory hits early
    out = None
    with ctx:
        for s, e_ in _chunk_spans(lens, order, chunk, chunk_tokens):
            sel = order[s:e_]
            L = int(lens[sel].max().item())                # trim to THIS chunk's longest
            e = model(ids[sel, :L], am[sel, :L], pm[sel, :L])
            if out is None:
                out = e.new_zeros((n, e.shape[-1]))
            out = out.index_copy(0, sel, e) if not grad else out.index_copy(0, sel, e)
    return out, (ids, am, pm)


def _mixed_lane_loss(scores, grades, soft, lanes, cfg, use_strata=None, cand_mask=None,
                     lane_counts=None):
    """Pairwise rank loss on biomedical rows, listwise KL on general rows:

        L = mean_over_biomed(pairwise)  +  cfg.kl_weight * mean_over_general(KL)

    Each lane is averaged over its own row count, so the size difference between the lanes does
    not re-weight the two terms. `lanes` is (B,) with 0 = biomedical (grades used), 1 = general
    (soft used).

    Returns (loss, pair_mean, kl_mean, n_pair, n_kl) so the two terms can be logged separately.
    """
    B = scores.shape[0]
    pair_terms, kl_terms = [], []
    for b in range(B):
        if int(lanes[b]) == 1:
            # Padded slots are set to -inf before the softmax so they take no probability mass
            # and the real candidates' distribution sums to 1.
            m = None if cand_mask is None else cand_mask[b]
            sl = scores[b].float() / cfg.tau
            tl = soft[b].float() / cfg.tau_teacher
            if m is not None:
                sl = sl.masked_fill(~m, float("-inf"))
                tl = tl.masked_fill(~m, float("-inf"))
            tgt = F.softmax(tl, dim=0)
            logp = F.log_softmax(sl, dim=0)
            logt = F.log_softmax(tl, dim=0)
            # KL(teacher || student), not cross-entropy: same gradient, but the logged value is 0
            # at step 0 and measures drift from the anchor directly. nan_to_num maps the
            # -inf * 0 of padded slots to 0.
            kl_terms.append((tgt * (torch.nan_to_num(logt, neginf=0.0)
                                    - torch.nan_to_num(logp, neginf=0.0))).sum())
        else:
            # Slice to the valid candidates rather than relying on NaN grades: the position
            # discount uses the rank of every score present, so padded slots would shift it.
            m = None if cand_mask is None else cand_mask[b]
            sc = scores[b] if m is None else scores[b][m]
            gr = grades[b] if m is None else grades[b][m]
            st = None if use_strata is None else (use_strata[b] if m is None else use_strata[b][m])
            pair_terms.append(pairwise_rank_loss(
                sc, gr, sigma=cfg.sigma, gap_power=cfg.gap_power, gain=cfg.gain,
                position_alpha=cfg.position_alpha, normalize=cfg.normalize, strata=st))
    z = scores.new_zeros(())
    if lane_counts is None:
        # single process: divide by what this batch actually holds
        n_pair, n_kl = len(pair_terms), len(kl_terms)
    else:
        # Multi-GPU: divide by the GLOBAL per-lane counts. Each rank contributes its own sum over
        # the global denominator, so the sum of the ranks' losses (and, via
        # allreduce_param_grads, their gradients) equals the single-process value.
        n_pair, n_kl = lane_counts
    pair_sum = torch.stack(pair_terms).sum() if pair_terms else z
    kl_sum = torch.stack(kl_terms).sum() if kl_terms else z
    pair_mean = pair_sum / max(n_pair, 1)
    kl_mean = kl_sum / max(n_kl, 1)
    return (pair_mean + cfg.kl_weight * kl_mean, pair_mean.detach(), kl_mean.detach(),
            len(pair_terms), len(kl_terms))


def _pool_loss(qc, dc, grades, cfg, strata=None, denom=None, lanes=None, soft=None,
               cand_mask=None, lane_counts=None):
    """(B,d) queries x (B*K,d) docs -> cosine scores (B,K) -> mixed-lane loss.

    Under multi-GPU the caller must pass `lane_counts` (global per-lane counts); otherwise each
    rank would divide by its own counts and the ranks' losses would not sum to the
    single-process loss.
    """
    B, K = grades.shape
    q = F.normalize(qc.float(), dim=-1)
    d = F.normalize(dc.float(), dim=-1).view(B, K, -1)
    scores = torch.einsum("bd,bkd->bk", q, d)          # cosine, single vector
    use_strata = strata if getattr(cfg, "pair_stratum_match", False) else None   # always None here
    if denom is not None and lane_counts is None:
        raise NotImplementedError(
            "mixed-lane loss under multi-GPU requires `lane_counts` (the GLOBAL per-lane query "
            "counts).")
    loss, pm, km, npair, nkl = _mixed_lane_loss(scores, grades, soft, lanes, cfg,
                                                use_strata, cand_mask, lane_counts)
    return loss, scores.detach(), {"pair": float(pm), "kl": float(km),
                                   "n_pair": npair, "n_kl": nkl}


def pairwise_grad_cache_step(model, batch, device, cfg, denom=None, lane_counts=None):
    """Two-pass gradient cache over {queries, K docs per query}. Fills param grads; returns loss."""
    chunk = cfg.chunk_size
    grades = batch["grades"].to(device)
    strata = batch["strata"].to(device) if "strata" in batch else None
    chunk_tokens = getattr(cfg, "chunk_tokens", 0)
    qc_raw, qdev = _forward_chunks(model, batch["q"], batch["q_pool"], device, chunk,
                                   grad=False, chunk_tokens=chunk_tokens)
    dc_raw, ddev = _forward_chunks(model, batch["d"], None, device, chunk,
                                   grad=False, chunk_tokens=chunk_tokens)
    qc = qc_raw.detach().requires_grad_(True)
    dc = dc_raw.detach().requires_grad_(True)
    lanes = batch["lanes"].to(device) if "lanes" in batch else None
    soft = batch["soft"].to(device) if "soft" in batch else None
    cand_mask = batch["cand_mask"].to(device) if "cand_mask" in batch else None
    loss, scores, terms = _pool_loss(qc, dc, grades, cfg, strata, denom=denom,
                                     lanes=lanes, soft=soft, cand_mask=cand_mask,
                                     lane_counts=lane_counts)
    loss.backward()
    # Pass 2 uses the same length bucketing and the same spans as pass 1, so each chunk's
    # re-forward is numerically identical to the forward whose embeddings were cached, and each
    # embedding receives its own cached gradient.
    for enc_dev, cached in ((qdev, qc), (ddev, dc)):
        ids, am, pm = enc_dev
        g = cached.grad
        lens = am.sum(dim=1)
        order = torch.argsort(lens, descending=True)
        for s, e_ in _chunk_spans(lens, order, chunk, chunk_tokens):
            sel = order[s:e_]
            L = int(lens[sel].max().item())
            e = model(ids[sel, :L], am[sel, :L], pm[sel, :L])
            torch.autograd.backward(e, grad_tensors=g[sel])
    return loss.detach(), scores, grades, terms


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #
def ddp_env():
    """(is_dist, rank, world_size, local_rank) from the torchrun-set environment."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return ws > 1, rank, ws, local_rank


def setup_dist(backend="nccl"):
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def allreduce_param_grads(params, world_size, group):
    """Sum each rank's local parameter gradients across ranks. No division: the loss is already
    normalised by the global per-lane counts. Identity on one process."""
    if world_size == 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)


def validate_cfg(cfg: PairwiseConfig, tok) -> None:
    """Refuse configurations that would run but train the wrong model."""
    if cfg.sigma <= 0:
        raise SystemExit(
            f"[pairwise] FATAL: sigma={cfg.sigma}. sigma is the loss's only scale parameter and is "
            f"per-model (2/median(|s_i-s_j|) over cross-grade pairs of this encoder's scores).")
    last_tok = cfg.pooling in ("eos", "last")
    if last_tok and cfg.mask_instruction:
        raise SystemExit(
            "[pairwise] FATAL: pooling='%s' with mask_instruction=True. Last-token pooling would "
            "pool a mid-query token. Set mask_instruction=false." % cfg.pooling)
    if cfg.base_kind == "qwen3" and not last_tok:
        raise SystemExit(f"[pairwise] FATAL: base_kind='qwen3' is last-token pooled; "
                         f"got pooling={cfg.pooling!r}.")
    if cfg.base_kind == "qwen3" and cfg.attention != "causal":
        raise SystemExit("[pairwise] FATAL: base_kind='qwen3' must set attention='causal'.")
    if getattr(tok, "padding_side", "right") != "right":
        raise SystemExit(f"[pairwise] FATAL: padding_side={tok.padding_side!r}; right padding only.")
    # Check on the real tokenizer how many special tokens trail a sequence, and log it.
    ids = tok("probe text", add_special_tokens=True)["input_ids"]
    n_trailing_eos = 0
    for t in reversed(ids):
        if t == tok.eos_token_id or t == tok.pad_token_id:
            n_trailing_eos += 1
        else:
            break
    print(f"[pairwise] tokenizer={type(tok).__name__} padding_side={tok.padding_side} "
          f"eos_id={tok.eos_token_id} pad_id={tok.pad_token_id} "
          f"trailing_special_tokens={n_trailing_eos} last_id={ids[-1]} "
          f"pooling={cfg.pooling} mask_instruction={cfg.mask_instruction}", flush=True)
    if last_tok and n_trailing_eos > 1:
        raise SystemExit(f"[pairwise] FATAL: tokenizer appends {n_trailing_eos} trailing special "
                         f"tokens; last-token pooling would read the LAST of them.")


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_ddp(cfg: PairwiseConfig):
    """Training on one process (plain python) or N ranks under torchrun; same code path.

    Every rank seeds identically before the model is built, so LoRA initialisation is identical
    on all ranks and no parameter broadcast is needed. Gradients are summed across ranks with
    `allreduce_param_grads` after each step's gradient-cache passes.
    """
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from model import NVEmbedBuildConfig, build_trainable_nvembed

    is_dist, rank, world_size, local_rank = ddp_env()
    assert cfg.lora_dropout == 0.0, "gradient cache requires lora_dropout=0 (pass-1/pass-2 must match)"
    assert cfg.batch_queries % world_size == 0, (
        f"batch_queries {cfg.batch_queries} not divisible by world_size {world_size}")
    B_local = cfg.batch_queries // world_size

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if is_dist:
        setup_dist("nccl" if torch.cuda.is_available() else "gloo")
    group = dist.group.WORLD if is_dist else None
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    is_main = (rank == 0)

    if is_main:
        os.makedirs(cfg.out_dir, exist_ok=True)
    if is_dist:
        dist.barrier()
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    logf = open(os.path.join(cfg.out_dir, "train_log.jsonl"), "a") if is_main else None

    def log(msg, **kv):
        if not is_main:
            return
        line = {"t": round(time.time(), 2), "msg": msg, **kv}
        print(f"[pairwise-ddp] {msg} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)
        logf.write(json.dumps(line) + "\n"); logf.flush()

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    validate_cfg(cfg, tok)

    train_ds = MixedBatchDataset(cfg.mixed_batches)
    if train_ds.batch_queries != cfg.batch_queries:
        # the batch file, not the config, sets the global batch size
        log("mixed batch_queries taken from the FILE", config=cfg.batch_queries,
            file=train_ds.batch_queries)
        cfg.batch_queries = train_ds.batch_queries
        assert cfg.batch_queries % world_size == 0, (
            f"mixed batch_queries {cfg.batch_queries} (from {cfg.mixed_batches}) not "
            f"divisible by world_size {world_size}")
        B_local = cfg.batch_queries // world_size
    log("data ready", n_train=len(train_ds),
        world_size=world_size, batch_queries=cfg.batch_queries, per_rank_queries=B_local)

    steps_per_epoch = max(1, len(train_ds) // cfg.batch_queries)
    total_steps = cfg.max_steps
    warmup = max(1, int(total_steps * cfg.warmup_frac))
    log("schedule", steps_per_epoch=steps_per_epoch, total_steps=total_steps, warmup=warmup,
        lr=cfg.lr)

    mcfg = NVEmbedBuildConfig(base_model=cfg.base_model, base_kind=cfg.base_kind,
                              pooling=cfg.pooling, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                              lora_dropout=cfg.lora_dropout, dtype=cfg.dtype,
                              gradient_checkpointing=cfg.gradient_checkpointing,
                              attention=cfg.attention,
                              attn_implementation=cfg.attn_implementation,
                              expected_trainable_params=cfg.expected_trainable_params)
    model = build_trainable_nvembed(mcfg).to(device)     # identical seed => identical weights
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    log("model ready", **model.trainable_parameter_count())

    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)
    if is_main:
        json.dump(asdict(cfg), open(os.path.join(cfg.out_dir, "train_config.json"), "w"), indent=2)

    step, epoch, t0 = 0, 0, time.time()

    # Each global batch is one pre-built batch replayed in file order; the ranks split its
    # queries into contiguous slices, so every step sees exactly the batch the single-process
    # run would see. If max_steps exceeds the number of batches, the file is replayed again.
    n_units = len(train_ds.batches)
    while step < total_steps:
        for s in range(0, n_units):
            if step >= total_steps:
                break
            gidx = train_ds.batch_indices(s)
            idx = gidx[rank * B_local:(rank + 1) * B_local]       # this rank's contiguous stride
            # global lane counts, identical on every rank
            lane_counts = train_ds.lane_counts(s)
            batch = collate([train_ds.example(i, epoch) for i in idx],
                            tok, cfg.max_len, cfg.instruction,
                            mask_instruction=cfg.mask_instruction,
                            pad_depth=True)
            opt.zero_grad(set_to_none=True)
            loss, _, _, terms = pairwise_grad_cache_step(model, batch, device, cfg,
                                                  denom=cfg.batch_queries,
                                                  lane_counts=lane_counts)
            allreduce_param_grads(params, world_size, group)      # local shard -> global gradient
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)  # AFTER the reduce: global norm
            opt.step(); sched.step(); step += 1

            if step % cfg.log_every == 0:
                lt = loss.detach().clone()
                if is_dist:
                    dist.all_reduce(lt, op=dist.ReduceOp.SUM, group=group)   # == single-process loss
                extra = {}
                if terms:
                    # Each rank holds its lane sums over the global denominators, so the two
                    # terms reduce by SUM exactly as the loss does.
                    pk = torch.tensor([terms["pair"], terms["kl"]], device=device,
                                      dtype=torch.float32)
                    if is_dist:
                        dist.all_reduce(pk, op=dist.ReduceOp.SUM, group=group)
                    extra = {"pair": round(float(pk[0]), 5), "kl": round(float(pk[1]), 5),
                             "kl_w": cfg.kl_weight,
                             "n_pair": lane_counts[0], "n_kl": lane_counts[1]}
                log("train", step=step, epoch=epoch, loss=round(lt.item(), 5),
                    lr=f"{sched.get_last_lr()[0]:.2e}",
                    vram=round(torch.cuda.max_memory_allocated() / 2**30, 1)
                    if torch.cuda.is_available() else 0.0,
                    s_per_step=round((time.time() - t0) / step, 1), **extra)
            if step % cfg.save_every == 0 or step == total_steps:
                d = os.path.join(cfg.out_dir, f"step_{step}")
                if is_main:
                    os.makedirs(d, exist_ok=True)
                    model.embedding_model.save_pretrained(os.path.join(d, "lora"))
                    json.dump(asdict(cfg), open(os.path.join(d, "train_config.json"), "w"), indent=2)
                    log("saved", step=step, path=d)
                if is_dist:
                    dist.barrier()
        epoch += 1
    log("done", steps=step, minutes=round((time.time() - t0) / 60, 1))
    cleanup_dist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    # optional per-field overrides, e.g. --max_steps 2 --out_dir runs/smoke
    types = {"str": str, "int": int, "float": float, "bool": lambda s: bool(int(s))}
    for f in dataclasses.fields(PairwiseConfig):
        ap.add_argument(f"--{f.name}", type=types[f.type], default=None)
    a = ap.parse_args()
    cfg = PairwiseConfig()
    names = {f.name for f in dataclasses.fields(PairwiseConfig)}
    unused = []
    for k, v in json.load(open(a.config)).items():
        if not k.startswith("_"):
            setattr(cfg, k, v)
            if k not in names:
                unused.append(k)
    if unused:
        print(f"[config] keys not read by this script: {sorted(unused)}", flush=True)
    for k, v in vars(a).items():
        if k != "config" and v is not None:
            setattr(cfg, k, v)
    missing = [k for k, v in asdict(cfg).items() if v is None]
    if missing:
        raise SystemExit(f"[config] missing required keys: {missing}")
    train_ddp(cfg)


if __name__ == "__main__":
    main()
