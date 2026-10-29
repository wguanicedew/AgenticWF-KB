"""
Optuna warm start from the KB (docs/hpo-warm-start.md, Python client). Install with the `hpo`
extra (Optuna).

    warm_start(kb, study, knowledge_base_ids=["alice:hpo-history"], description="…",
               search_space=SPACE, n=10)          # POST /records/suggest → study.enqueue_trial
    study.optimize(objective, n_trials=100)
    upload_study(kb, study, kb_id="alice:hpo-history", job_id=RUN_ID, search_space=SPACE)
                                                  # POST /knowledge-bases/{kb_id}/records (hpo_job)
"""

from __future__ import annotations

import datetime
import math
from typing import Any

from agenticwf.kb.client import KBClient


def space_from_distributions(distributions: dict) -> dict:
    """An hpo_job search_space from Optuna distributions (FloatDistribution, IntDistribution,
    CategoricalDistribution)."""
    space = {}
    for name, dist in distributions.items():
        kind = type(dist).__name__
        if kind in ("FloatDistribution", "UniformDistribution", "LogUniformDistribution", "DiscreteUniformDistribution"):
            spec = {"type": "float", "low": dist.low, "high": dist.high, "log": bool(getattr(dist, "log", False))}
            if getattr(dist, "step", None):
                spec["step"] = dist.step
        elif kind in ("IntDistribution", "IntUniformDistribution", "IntLogUniformDistribution"):
            spec = {"type": "int", "low": dist.low, "high": dist.high, "log": bool(getattr(dist, "log", False))}
            if getattr(dist, "step", 1) not in (None, 1):
                spec["step"] = dist.step
        elif kind == "CategoricalDistribution":
            spec = {"type": "categorical", "choices": list(dist.choices)}
        else:
            raise ValueError(f"unsupported distribution for {name}: {kind}")
        space[name] = spec
    return space


def _space(study, search_space: dict | None) -> dict:
    if search_space:
        return search_space
    trials = [t for t in study.trials if t.distributions]
    if not trials:
        raise ValueError("give search_space (the study has no trials to read distributions from)")
    return space_from_distributions(trials[-1].distributions)


def _direction(study) -> str:
    return "maximize" if study.direction.name == "MAXIMIZE" else "minimize"


def warm_start(kb: KBClient, study, search_space: dict, knowledge_base_ids: list[str] | None = None,
               objective_name: str = "objective", description: str = "", n: int = 10, n_total: int | None = None,
               include_history: bool = False, filter: dict | None = None, options: dict | None = None) -> dict:
    """Enqueue the KB's suggested initial points into `study` (and, with include_history, add the
    identical-space history as completed trials). Returns the suggest response."""
    body: dict[str, Any] = {"objective": {"name": objective_name, "direction": _direction(study)},
                            "search_space": search_space, "description": description, "n": n,
                            "n_total": n_total or n, "options": {**(options or {}), "include_history": include_history}}
    if knowledge_base_ids:
        body["knowledge_base_ids"] = knowledge_base_ids
    if filter:
        body["filter"] = filter
    data = kb.suggest(**body)
    for point in data["points"]:
        study.enqueue_trial(point["params"], skip_if_exists=True)
    if include_history and data.get("history"):
        import optuna

        distributions = _distributions(search_space)
        for past in data["history"]:
            if past.get("value") is None:
                continue
            study.add_trial(optuna.trial.create_trial(params=past["params"], distributions=distributions,
                                                      value=past["value"]))
    return data


def _distributions(space: dict) -> dict:
    import optuna

    out = {}
    for name, spec in space.items():
        if spec["type"] == "categorical":
            out[name] = optuna.distributions.CategoricalDistribution(spec["choices"])
        elif spec["type"] == "int":
            out[name] = optuna.distributions.IntDistribution(int(spec["low"]), int(spec["high"]),
                                                             log=bool(spec.get("log")), step=int(spec.get("step") or 1))
        else:
            out[name] = optuna.distributions.FloatDistribution(spec["low"], spec["high"], log=bool(spec.get("log")),
                                                               step=spec.get("step"))
    return out


def study_record(study, job_id: str, search_space: dict | None = None, objective_name: str = "objective",
                 description: str = "", metadata: dict | None = None, record_name: str | None = None) -> dict:
    """An hpo_job record for `study`."""
    status = {"COMPLETE": "complete", "PRUNED": "pruned", "FAIL": "fail", "RUNNING": "running", "WAITING": "running"}
    trials = []
    for t in study.trials:
        value = t.value if t.value is not None and math.isfinite(t.value) else None
        trials.append({"number": t.number, "params": t.params, "value": value,
                       "status": status.get(t.state.name, "fail") if value is not None or t.state.name != "COMPLETE"
                       else "fail"})
    return {"job_id": job_id, "record_name": record_name or job_id, "description": description,
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "objective": {"name": objective_name, "direction": _direction(study)},
            "search_space": _space(study, search_space), "trials": trials, "metadata": metadata or {}}


def upload_study(kb: KBClient, study, kb_id: str, job_id: str, search_space: dict | None = None,
                 objective_name: str = "objective", description: str = "", metadata: dict | None = None) -> dict:
    record = study_record(study, job_id, search_space, objective_name, description, metadata)
    return kb.upsert_records(kb_id, [record], record_type="hpo_job")
