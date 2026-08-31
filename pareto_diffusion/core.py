"""Dependency-light Pareto ranking and evaluation utilities (all objectives maximize)."""

from __future__ import annotations

import numpy as np


OBJECTIVE_NAMES = (
    "disturbance_resistance",
    "contact_coverage",
    "angular_span",
)


def _matrix(values, name="objectives"):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(OBJECTIVE_NAMES):
        raise ValueError(f"{name} must have shape (N, {len(OBJECTIVE_NAMES)})")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def dominates(a, b, atol=1e-12):
    """Whether maximization point a Pareto-dominates b."""
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a >= b - atol) and np.any(a > b + atol))


def non_dominated_sort(objectives, feasible=None, violation=None):
    """NSGA-II ranks with feasibility-first constrained dominance."""
    values = _matrix(objectives)
    n = len(values)
    feasible = np.ones(n, dtype=bool) if feasible is None else np.asarray(feasible, dtype=bool)
    violation = np.zeros(n) if violation is None else np.asarray(violation, dtype=np.float64)
    if feasible.shape != (n,) or violation.shape != (n,):
        raise ValueError("feasible and violation must have shape (N,)")

    def preferred(i, j):
        if feasible[i] != feasible[j]:
            return feasible[i]
        if not feasible[i] and not feasible[j]:
            return violation[i] < violation[j] - 1e-12
        return dominates(values[i], values[j])

    dominates_set = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            i_better, j_better = preferred(i, j), preferred(j, i)
            if i_better and not j_better:
                dominates_set[i].append(j); dominated_count[j] += 1
            elif j_better and not i_better:
                dominates_set[j].append(i); dominated_count[i] += 1
    ranks = np.full(n, -1, dtype=np.int64)
    front = np.flatnonzero(dominated_count == 0).tolist()
    rank = 0
    while front:
        next_front = []
        for i in front:
            ranks[i] = rank
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        front, rank = next_front, rank + 1
    if np.any(ranks < 0):
        raise RuntimeError("Non-dominated sorting failed to rank all points")
    return ranks


def crowding_distance(objectives, ranks):
    values = _matrix(objectives)
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.shape != (len(values),):
        raise ValueError("ranks must have shape (N,)")
    distance = np.zeros(len(values), dtype=np.float64)
    for rank in np.unique(ranks):
        ids = np.flatnonzero(ranks == rank)
        if len(ids) <= 2:
            distance[ids] = np.inf
            continue
        for objective in range(values.shape[1]):
            order = ids[np.argsort(values[ids, objective], kind="stable")]
            distance[order[[0, -1]]] = np.inf
            span = values[order[-1], objective] - values[order[0], objective]
            if span > 1e-12:
                interior = order[1:-1]
                distance[interior] += (
                    values[order[2:], objective] - values[order[:-2], objective]
                ) / span
    return distance


def preference(i, j, ranks, crowding, atol=1e-12):
    """Return 1 if i is preferred, -1 if j is preferred, 0 for an unresolved tie."""
    if ranks[i] != ranks[j]:
        return 1 if ranks[i] < ranks[j] else -1
    ci, cj = crowding[i], crowding[j]
    if np.isinf(ci) != np.isinf(cj):
        return 1 if np.isinf(ci) else -1
    if np.isfinite(ci) and abs(ci - cj) > atol:
        return 1 if ci > cj else -1
    return 0


def build_preference_pairs(ranks, crowding, max_pairs=None, seed=0):
    """Create balanced ordered pairs; label 1 always means the first item is preferred."""
    ranks, crowding = np.asarray(ranks), np.asarray(crowding)
    pairs = []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            result = preference(i, j, ranks, crowding)
            if result:
                winner, loser = (i, j) if result > 0 else (j, i)
                pairs.extend(((winner, loser, 1.0), (loser, winner, 0.0)))
    if not pairs:
        raise ValueError("No resolvable preference pairs were produced")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    if max_pairs is not None:
        order = order[: min(int(max_pairs), len(order))]
    return np.asarray([pairs[i] for i in order], dtype=np.float64)


def pareto_front(objectives, feasible=None, violation=None):
    ranks = non_dominated_sort(objectives, feasible, violation)
    return np.flatnonzero(ranks == 0)


def hypervolume(objectives, reference=(0.0, 0.0, 0.0)):
    """Exact dominated hypervolume for three maximization objectives."""
    points = _matrix(objectives)
    ref = np.asarray(reference, dtype=np.float64)
    if ref.shape != (3,):
        raise ValueError("reference must have shape (3,)")
    points = points[np.all(points > ref, axis=1)]
    if not len(points):
        return 0.0

    def recursive(pts, origin):
        dimensions = pts.shape[1]
        if dimensions == 1:
            return max(float(pts[:, 0].max() - origin[0]), 0.0)
        boundaries = np.unique(np.concatenate(([origin[0]], pts[:, 0])))
        total = 0.0
        for low, high in zip(boundaries[:-1], boundaries[1:]):
            active = pts[pts[:, 0] >= high - 1e-12]
            if len(active):
                total += (high - low) * recursive(active[:, 1:], origin[1:])
        return total

    return float(recursive(points, ref))


def inverted_generational_distance(objectives, reference_front):
    points, reference = _matrix(objectives), _matrix(reference_front, "reference_front")
    distances = np.linalg.norm(reference[:, None, :] - points[None, :, :], axis=-1)
    return float(distances.min(axis=1).mean())


def spacing(objectives):
    points = _matrix(objectives)
    if len(points) < 2:
        return 0.0
    distance = np.abs(points[:, None, :] - points[None, :, :]).sum(axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min(axis=1).std(ddof=1)) if len(points) > 2 else 0.0


def summarize_front(objectives, reference=(0.0, 0.0, 0.0), reference_front=None):
    values = _matrix(objectives)
    ids = pareto_front(values)
    front = values[ids]
    result = {
        "num_designs": len(values),
        "num_non_dominated": len(ids),
        "non_dominated_fraction": len(ids) / max(len(values), 1),
        "hypervolume": hypervolume(front, reference),
        "spacing": spacing(front),
        "objective_min": front.min(axis=0).tolist(),
        "objective_max": front.max(axis=0).tolist(),
    }
    if reference_front is not None:
        result["igd"] = inverted_generational_distance(front, reference_front)
    return result, ids
