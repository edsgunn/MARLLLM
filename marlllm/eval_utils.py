"""
Evaluation utilities for the Lewis-2017 deal-or-no-deal negotiation task.

These helpers are pure functions over the per-episode item counts and private
utility vectors — they make no assumptions about how an agent generated its
allocation.  Used by ``evaluate_negotiation.py`` to compute capability metrics
(joint utility, Pareto efficiency, etc.) without coupling them to the trainer.

Pareto frontier
---------------
For DealOrNoDealEnv each item type has at most ``max_item_count`` units (5 by
default), and there are 3 item types — so there are at most ``6^3 = 216``
possible per-agent allocations.  We enumerate them exhaustively, compute the
joint score for every valid split, and filter to the Pareto frontier.  This is
cheap enough to run on every evaluation episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class NegotiationOutcome:
    """The capability-relevant fields of a single negotiation episode."""
    deal: bool
    score_a: int                 # focal-agent score (0 if no-deal)
    score_b: int                 # other-agent score (0 if no-deal)
    items: tuple[int, ...]
    values_a: tuple[int, ...]
    values_b: tuple[int, ...]


def all_valid_splits(items: tuple[int, ...]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate every valid (alloc_a, alloc_b) split where alloc_a + alloc_b == items."""
    ranges = [range(c + 1) for c in items]
    splits = []
    for alloc_a in product(*ranges):
        alloc_b = tuple(items[i] - alloc_a[i] for i in range(len(items)))
        splits.append((alloc_a, alloc_b))
    return splits


def score(alloc: tuple[int, ...], values: tuple[int, ...]) -> int:
    return sum(a * v for a, v in zip(alloc, values))


def pareto_frontier(
    items: tuple[int, ...],
    values_a: tuple[int, ...],
    values_b: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Return the (score_a, score_b) Pareto frontier of all valid splits."""
    points: set[tuple[int, int]] = set()
    for alloc_a, alloc_b in all_valid_splits(items):
        points.add((score(alloc_a, values_a), score(alloc_b, values_b)))

    frontier: list[tuple[int, int]] = []
    pts = sorted(points)
    for sa, sb in pts:
        # Strictly dominated if any other point has >= in both and > in one.
        dominated = any(
            (oa >= sa and ob >= sb) and (oa > sa or ob > sb)
            for oa, ob in points
        )
        if not dominated:
            frontier.append((sa, sb))
    return sorted(frontier)


def max_joint_score(items: tuple[int, ...], values_a, values_b) -> int:
    """Maximum achievable joint utility (a benchmark for Pareto efficiency)."""
    return max(
        score(a, values_a) + score(b, values_b)
        for a, b in all_valid_splits(items)
    )


def pareto_efficiency(outcome: NegotiationOutcome) -> float:
    """
    Joint utility of the realised split divided by the max joint utility achievable
    given the two agents' private values.  ``0.0`` for no-deals.

    Returns a value in [0, 1] where 1.0 means the split sits on the joint-utility
    optimum.  Distinct from "is the realised point on the Pareto frontier?" — that
    is captured by :func:`is_pareto_optimal`.
    """
    if not outcome.deal:
        return 0.0
    best = max_joint_score(outcome.items, outcome.values_a, outcome.values_b)
    if best == 0:
        return 0.0
    realised = outcome.score_a + outcome.score_b
    return realised / best


def is_pareto_optimal(outcome: NegotiationOutcome) -> bool:
    """True iff (score_a, score_b) lies on the Pareto frontier."""
    if not outcome.deal:
        return False
    frontier = pareto_frontier(outcome.items, outcome.values_a, outcome.values_b)
    return (outcome.score_a, outcome.score_b) in frontier


def summarise(outcomes: list[NegotiationOutcome]) -> dict:
    """
    Aggregate a list of episode outcomes into the capability metrics defined in
    Phase A §3.1: joint utility, individual utility, agreement rate, Pareto
    efficiency.  Returns a flat dict ready for JSON serialisation.
    """
    n = len(outcomes)
    if n == 0:
        return {"n_episodes": 0}

    deals = [o for o in outcomes if o.deal]
    n_deals = len(deals)
    sum_joint  = sum(o.score_a + o.score_b for o in deals)
    sum_a      = sum(o.score_a for o in deals)
    sum_b      = sum(o.score_b for o in deals)

    pareto_eff = [pareto_efficiency(o) for o in outcomes]
    pareto_opt = [1.0 if is_pareto_optimal(o) else 0.0 for o in outcomes]

    return {
        "n_episodes":            n,
        "agreement_rate":        n_deals / n,
        "joint_utility_mean":    (sum_joint / n_deals) if n_deals else 0.0,
        "joint_utility_mean_all":sum_joint / n,        # zeros counted for no-deals
        "individual_utility_a":  (sum_a / n_deals) if n_deals else 0.0,
        "individual_utility_b":  (sum_b / n_deals) if n_deals else 0.0,
        "individual_utility_a_all": sum_a / n,
        "individual_utility_b_all": sum_b / n,
        "pareto_efficiency_mean":  sum(pareto_eff) / n,
        "pareto_optimal_rate":     sum(pareto_opt) / n,
    }
