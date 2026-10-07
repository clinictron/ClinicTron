"""Weighted pairwise ranking loss over teacher-graded candidate pools."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _position_weight(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-document position weight pi(r) = (1/r)^alpha, r = 1-based rank by score, descending.

    Computed on detached scores: the rank is piecewise constant in the scores, so it carries no
    useful gradient (as in LambdaRank's |dNDCG| weight). alpha=0 returns all ones.
    """
    if alpha == 0.0:
        return torch.ones_like(scores)
    order = torch.argsort(scores.detach(), descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device)
    return (1.0 / ranks.to(scores.dtype)) ** alpha


def pairwise_rank_loss(
    scores: torch.Tensor,
    grades: torch.Tensor,
    *,
    sigma: float = 16.0,
    gap_power: float = 1.0,
    gain: str = "linear",
    position_alpha: float = 0.25,
    normalize: str = "weighted_mean",
    strata: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted pairwise logistic (RankNet-style) ranking loss over one query's candidates.

        L = (1/Z) * sum_{i,j : g_i > g_j} w_ij * log(1 + exp(-sigma * (s_i - s_j)))
        w_ij = |g_i - g_j| * max(pi(r_i), pi(r_j)),   Z = sum w_ij

    ``scores``  (K,) student scores (cosine similarity of L2-normalised embeddings).
    ``grades``  (K,) teacher grades; NaN entries are excluded from every pair.
    ``strata``  (K,) optional integer group id per candidate. When given, only candidates in the
                same stratum form a pair, so any document property that defines the stratum is
                constant within every comparison and cannot be used to separate grades.

    Only strictly unequal grades form a pair: equal grades are unconstrained, and ungraded easy
    negatives (grade 0) sit below every graded candidate without being ordered among themselves.

    Supported settings: gap_power=1.0, gain="linear", normalize="weighted_mean".
    """
    if scores.shape != grades.shape or scores.dim() != 1:
        raise ValueError(f"scores {tuple(scores.shape)} / grades {tuple(grades.shape)} must be equal-length 1-D")
    if gap_power != 1.0:
        raise ValueError(f"unsupported gap_power {gap_power!r}; supported: 1.0")
    if gain != "linear":
        raise ValueError(f"unsupported gain {gain!r}; supported: 'linear'")
    if normalize != "weighted_mean":
        raise ValueError(f"unsupported normalize {normalize!r}; supported: 'weighted_mean'")
    valid = ~torch.isnan(grades)
    g = torch.nan_to_num(grades, nan=0.0)
    gain_v = g

    pair = (g.unsqueeze(1) > g.unsqueeze(0)) & valid.unsqueeze(1) & valid.unsqueeze(0)
    if strata is not None:
        if strata.shape != grades.shape:
            raise ValueError(f"strata {tuple(strata.shape)} must match grades {tuple(grades.shape)}")
        pair = pair & (strata.unsqueeze(1) == strata.unsqueeze(0))
    if not bool(pair.any()):
        return scores.sum() * 0.0  # no constraint -> exactly zero loss AND zero gradient

    w = (gain_v.unsqueeze(1) - gain_v.unsqueeze(0)).abs()
    pi = _position_weight(scores, position_alpha)
    w = w * torch.maximum(pi.unsqueeze(1), pi.unsqueeze(0))
    w = w * pair

    diff = scores.unsqueeze(1) - scores.unsqueeze(0)          # diff[i,j] = s_i - s_j
    per_pair = F.softplus(-sigma * diff)                      # = log(1 + exp(-sigma*(s_i-s_j)))
    total = (w * per_pair).sum()

    z = w.sum().clamp_min(1e-12)
    return total / z
