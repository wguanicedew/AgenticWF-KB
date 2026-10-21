"""
The warm-start algorithm (docs/hpo-warm-start.md, Algorithm), deterministic for a given seed.
No database and no model: the plugin feeds it candidate jobs and gets points back.

    job_similarity      S = wp·P + wr·R + wt·T, × min(1, n_complete/10)^0.5
    transform           a source trial into the new space (clip, snap, nearest choice, fill)
    encode / dedup      unit-cube encoding (log ranges, one-hot / √2 categoricals), merge < 0.02
    mmr                 λ·score − (1−λ)·max similarity to the selected points
    halton              seeded, scrambled Halton points mapped into the space
"""

from __future__ import annotations

import math
import random
import re

DEFAULTS = {"max_source_jobs": 5, "per_job": 5, "min_similarity": 0.3, "min_coverage": 0.5, "top_quantile": 0.2,
            "fill_strategy": "default_or_center", "include_history": False, "seed": 0}
WEIGHTS = {"params": 0.4, "ranges": 0.3, "task": 0.3}
MMR_LAMBDA = 0.7
DEDUP_RADIUS = 0.02
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
          103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199)


# --- names ---------------------------------------------------------------------------------------------
def snake(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def alias_map(aliases: dict) -> dict[str, str]:
    """{alias (snake): canonical} from {canonical: [aliases]}."""
    out = {}
    for canonical, names in (aliases or {}).items():
        for alias in [names] if isinstance(names, str) else names:
            out[snake(alias)] = snake(canonical)
    return out


def canonical_name(name: str, aliases: dict[str, str]) -> str:
    key = snake(name)
    return aliases.get(key, key)


# --- percentiles -------------------------------------------------------------------------------------------
def percentiles(values: list[float | None], direction: str) -> list[float | None]:
    """Direction-aware percentile of each value among the numeric ones (1.0 best, 0.0 worst)."""
    numeric = sorted({v for v in values if v is not None})
    if not numeric:
        return [None for _ in values]
    if direction == "maximize":
        numeric = numeric[::-1]
    rank = {v: i for i, v in enumerate(numeric)}
    n = len(numeric)
    return [None if v is None else (1.0 if n == 1 else 1.0 - rank[v] / (n - 1)) for v in values]


# --- job similarity ------------------------------------------------------------------------------------------
def _numeric_range(spec: dict) -> tuple[float, float] | None:
    if spec.get("type") in ("float", "int") and spec.get("low") is not None and spec.get("high") is not None:
        return float(spec["low"]), float(spec["high"])
    return None


def range_overlap(a: dict, b: dict) -> float:
    """Overlap ÷ union of two parameter ranges (log space when either is log); categorical
    Jaccard; type mismatch 0."""
    ta, tb = a.get("type"), b.get("type")
    if ta == "categorical" and tb == "categorical":
        ca, cb = {str(x) for x in a.get("choices") or []}, {str(x) for x in b.get("choices") or []}
        return len(ca & cb) / len(ca | cb) if ca | cb else 0.0
    ra, rb = _numeric_range(a), _numeric_range(b)
    if ra is None or rb is None:
        return 0.0
    if a.get("log") or b.get("log"):
        if min(ra[0], rb[0]) <= 0:
            return 0.0
        ra, rb = (math.log(ra[0]), math.log(ra[1])), (math.log(rb[0]), math.log(rb[1]))
    union = max(ra[1], rb[1]) - min(ra[0], rb[0])
    overlap = max(0.0, min(ra[1], rb[1]) - max(ra[0], rb[0]))
    if union <= 0:
        return 1.0 if ra == rb else 0.0
    return overlap / union


def job_similarity(space: dict, other_space: dict, task: float | None, n_complete: int,
                   weights: dict | None = None) -> tuple[float, dict]:
    weights = {**WEIGHTS, **(weights or {})}
    names, other = set(space), set(other_space)
    p = len(names & other) / len(names | other) if names | other else 0.0
    shared = names & other
    r = sum(range_overlap(space[n], other_space[n]) for n in shared) / len(shared) if shared else 0.0
    parts = {"params": p, "ranges": r}
    if task is not None:
        parts["task"] = max(0.0, task)
    total = sum(weights[k] for k in parts)
    s = sum(weights[k] * v for k, v in parts.items()) / total if total else 0.0
    confidence = min(1.0, n_complete / 10) ** 0.5
    return s * confidence, {k: round(v, 4) for k, v in parts.items()}


# --- transforming points ---------------------------------------------------------------------------------------
def centre(spec: dict, rng: random.Random | None = None):
    if "default" in spec and spec["default"] is not None:
        return spec["default"]
    if spec.get("type") == "categorical":
        choices = spec.get("choices") or [None]
        return rng.choice(choices) if rng else choices[0]
    low, high = float(spec["low"]), float(spec["high"])
    if rng:
        if spec.get("log") and low > 0:
            value = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            value = rng.uniform(low, high)
    else:
        value = math.sqrt(low * high) if spec.get("log") and low > 0 else (low + high) / 2
    return snap(value, spec)


def snap(value: float, spec: dict):
    low, high = float(spec["low"]), float(spec["high"])
    value = min(high, max(low, float(value)))
    step = spec.get("step")
    if step:
        value = low + round((value - low) / float(step)) * float(step)
        value = min(high, max(low, value))
    if spec.get("type") == "int":
        return int(round(value))
    return float(value)


def transform(params: dict, space: dict, fill_strategy: str, rng: random.Random) -> tuple[dict, list, list, float]:
    """(point, filled, clipped, coverage) of a source trial in the new space."""
    point, filled, clipped = {}, [], []
    for name, spec in space.items():
        value = params.get(name)
        kind = spec.get("type")
        if value is not None and kind in ("float", "int") and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            new = snap(value, spec)
            if not math.isclose(float(new), float(value), rel_tol=1e-12, abs_tol=1e-15):
                clipped.append(name)
            point[name] = new
            continue
        if value is not None and kind == "categorical":
            choices = spec.get("choices") or []
            if value in choices:
                point[name] = value
                continue
            numbers = [c for c in choices if isinstance(c, (int, float)) and not isinstance(c, bool)]
            if numbers and isinstance(value, (int, float)) and not isinstance(value, bool):
                point[name] = min(numbers, key=lambda c: (abs(c - value), choices.index(c)))
                clipped.append(name)
                continue
        point[name] = centre(spec, rng if fill_strategy == "random" else None)
        filled.append(name)
    coverage = 1 - len(filled) / len(space) if space else 0.0
    return point, filled, clipped, coverage


# --- encoding, dedup, MMR -------------------------------------------------------------------------------------
def encode(point: dict, space: dict) -> list[float]:
    vector = []
    for name in sorted(space):
        spec, value = space[name], point.get(name)
        if spec.get("type") == "categorical":
            choices = spec.get("choices") or []
            vector += [(1.0 if value == c else 0.0) / math.sqrt(2) for c in choices]
            continue
        low, high = float(spec["low"]), float(spec["high"])
        v = float(value)
        if spec.get("log") and low > 0:
            low, high, v = math.log(low), math.log(high), math.log(max(v, 1e-300))
        vector.append(0.0 if high == low else (v - low) / (high - low))
    return vector


def distance(a: list[float], b: list[float]) -> float:
    if not a:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / math.sqrt(len(a))


def dedup(candidates: list[dict], space: dict, radius: float = DEDUP_RADIUS) -> list[dict]:
    kept: list[dict] = []
    for cand in sorted(candidates, key=lambda c: -c["score"]):
        vec = encode(cand["params"], space)
        if any(distance(vec, k["_vec"]) < radius for k in kept):
            continue
        kept.append({**cand, "_vec": vec})
    return kept


def mmr(candidates: list[dict], n: int, lam: float = MMR_LAMBDA) -> list[dict]:
    chosen, remaining = [], list(candidates)
    while remaining and len(chosen) < n:
        def value(c):
            redundancy = max((1 - distance(c["_vec"], s["_vec"]) for s in chosen), default=0.0)
            return lam * c["score"] - (1 - lam) * redundancy
        best = max(remaining, key=value)
        chosen.append(best)
        remaining.remove(best)
    return chosen


# --- Halton --------------------------------------------------------------------------------------------------
def _radical_inverse(index: int, base: int, perm: list[int]) -> float:
    result, f = 0.0, 1.0 / base
    while index > 0:
        result += perm[index % base] * f
        index //= base
        f /= base
    return result


def halton(space: dict, count: int, seed: int = 0, skip: int = 0) -> list[dict]:
    """Seeded, scrambled (random digit permutation per dimension) Halton points in the space."""
    names = sorted(space)
    if not names or count <= 0:
        return []
    rng = random.Random(seed * 7919 + 17)
    perms = []
    for d in range(len(names)):
        base = PRIMES[d % len(PRIMES)]
        perm = list(range(base))
        rest = perm[1:]
        rng.shuffle(rest)
        perms.append([0] + rest)
    offset = 1 + skip + rng.randrange(0, 64)
    points = []
    for k in range(count):
        point = {}
        for d, name in enumerate(names):
            u = _radical_inverse(offset + k, PRIMES[d % len(PRIMES)], perms[d])
            point[name] = from_unit(u, space[name])
        points.append(point)
    return points


def from_unit(u: float, spec: dict):
    if spec.get("type") == "categorical":
        choices = spec.get("choices") or [None]
        return choices[min(len(choices) - 1, int(u * len(choices)))]
    low, high = float(spec["low"]), float(spec["high"])
    if spec.get("log") and low > 0:
        value = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
    else:
        value = low + u * (high - low)
    return snap(value, spec)


# --- the whole suggestion -------------------------------------------------------------------------------------
def suggest(space: dict, objective: dict, jobs: list[dict], n: int, n_total: int | None = None,
            options: dict | None = None, weights: dict | None = None) -> dict:
    """`jobs`: candidate source jobs, each {record_id, knowledge_base_id, job_id, record_name,
    search_space (canonical), objective, trials [{params, value, status, percentile}], n_complete,
    task (text cosine or None)}. Returns the response's points, source_jobs, unmatched_params,
    history."""
    opts = {**DEFAULTS, **(options or {})}
    rng = random.Random(int(opts["seed"]))
    n = max(0, int(n))
    n_total = max(n, int(n_total or n))
    scored = []
    for job in jobs:
        s, breakdown = job_similarity(space, job["search_space"], job.get("task"), int(job.get("n_complete") or 0),
                                      weights)
        if s >= float(opts["min_similarity"]):
            scored.append((s, breakdown, job))
    scored.sort(key=lambda t: (-t[0], str(t[2].get("job_id"))))
    sources = scored[:int(opts["max_source_jobs"])]
    candidates = []
    for s, _, job in sources:
        trials = [t for t in job["trials"] if t.get("status", "complete") == "complete"
                  and t.get("percentile") is not None and t["percentile"] >= 1 - float(opts["top_quantile"])]
        trials.sort(key=lambda t: (-t["percentile"], t.get("index", 0)))
        for trial in trials[:int(opts["per_job"])]:
            point, filled, clipped, coverage = transform(trial.get("params") or {}, space, opts["fill_strategy"], rng)
            if coverage < float(opts["min_coverage"]):
                continue
            candidates.append({"params": point, "score": s * trial["percentile"], "filled_params": filled,
                               "clipped_params": clipped, "source": {
                                   "record_id": job["record_id"], "knowledge_base_id": job["knowledge_base_id"],
                                   "job_id": job["job_id"], "record_name": job.get("record_name"),
                                   "trial_index": trial.get("index"), "value": trial.get("value"),
                                   "percentile": round(trial["percentile"], 4), "job_similarity": round(s, 4)}})
    selected = mmr(dedup(candidates, space), n)
    points = [{k: v for k, v in c.items() if k != "_vec"} for c in selected]
    for p in points:
        p["score"] = round(p["score"], 4)
    fill = n_total - len(points)
    for point in halton(space, fill, int(opts["seed"])):
        points.append({"params": point, "source": {"type": "halton"}})
    covered = set()
    for _, _, job in sources:
        covered |= set(job["search_space"])
    history = None
    if opts["include_history"]:
        history = []
        for _, _, job in sources:
            if job["search_space"] == space and _same_objective(job.get("objective"), objective):
                history += [{"params": t["params"], "value": t.get("value"), "job_id": job["job_id"]}
                            for t in job["trials"] if t.get("status", "complete") == "complete"]
    return {"points": points,
            "source_jobs": [{"record_id": job["record_id"], "knowledge_base_id": job["knowledge_base_id"],
                             "job_id": job["job_id"], "record_name": job.get("record_name"), "similarity": round(s, 4),
                             "breakdown": b, "n_complete": job.get("n_complete")} for s, b, job in sources],
            "unmatched_params": sorted(set(space) - covered), "history": history}


def _same_objective(a: dict | None, b: dict | None) -> bool:
    a, b = a or {}, b or {}
    return snake(a.get("name", "")) == snake(b.get("name", "")) and a.get("direction") == b.get("direction")
