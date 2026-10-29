"""The hpo plugin (docs/hpo-warm-start.md): canonicalisation, percentiles, the warm-start
algorithm, the suggest operation end to end, access, and the synthetic warm-start benchmark."""

import math
import random
import statistics

import pytest

from conftest import Caller, drain, make_client, user
from agenticwf.kb.runtime.records.plugins.hpo import HPOPlugin
from agenticwf.kb.runtime.records.plugins.hpo import algorithm as alg

RW = ("app:access", "knowledge:read", "knowledge:write")
SPACE = {"learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
         "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
         "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
         "warmup_steps": {"type": "int", "low": 0, "high": 2000, "step": 100}}


# --- units ----------------------------------------------------------------------------------------------
def test_canonicalisation_and_derive():
    plugin = HPOPlugin()
    cfg = {"hpo": {"param_aliases": {"learning_rate": ["lr", "learningRate"]},
                   "objective_aliases": {"val_loss": ["validation_loss"]}}}
    record = plugin.normalize({"job_id": "j", "objective": {"name": "Validation Loss", "direction": "minimize"},
                               "search_space": {"learningRate": SPACE["learning_rate"], "BatchSize": SPACE["batch_size"]},
                               "trials": [{"params": {"lr": 0.01, "batch_size": 64}, "value": 0.5},
                                          {"params": {"lr": 0.001, "batch_size": 32}, "value": 0.3},
                                          {"params": {"lr": 0.1, "batch_size": 32}, "status": "fail"}]}, cfg)
    assert record["objective"]["name"] == "val_loss" and set(record["search_space"]) == {"learning_rate", "batch_size"}
    assert record["trials"][0]["params"] == {"learning_rate": 0.01, "batch_size": 64}
    assert plugin.validate(record, cfg) == []
    assert plugin.validate({**record, "trials": [{"params": {"momentum": 0.9}, "value": 1}]}, cfg)
    assert plugin.validate({**record, "search_space": {"x": {"type": "float", "low": 0, "high": 1, "log": True}}}, cfg)
    derived, items = plugin.derive(record)
    assert derived["best_value"] == 0.3 and derived["n_complete"] == 2
    assert [i["percentile"] for i in items] == [0.0, 1.0, None]
    assert alg.percentiles([0.9, 0.8, 0.7], "maximize") == [1.0, 0.5, 0.0]


def test_range_overlap_and_transform():
    assert alg.range_overlap({"type": "float", "low": 0, "high": 1}, {"type": "float", "low": 0.5, "high": 1.5}) \
        == pytest.approx(1 / 3)
    assert alg.range_overlap({"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
                             {"type": "float", "low": 1e-3, "high": 1e-1}) == pytest.approx(1 / 3)
    assert alg.range_overlap({"type": "categorical", "choices": [1, 2]}, {"type": "categorical", "choices": [2, 3]}) \
        == pytest.approx(1 / 3)
    assert alg.range_overlap({"type": "categorical", "choices": [1]}, {"type": "float", "low": 0, "high": 1}) == 0.0
    rng = random.Random(0)
    point, filled, clipped, coverage = alg.transform(
        {"learning_rate": 0.5, "batch_size": 100, "warmup_steps": 437, "old_param": 3}, SPACE, "default_or_center", rng)
    assert point["learning_rate"] == 0.1 and point["batch_size"] == 128 and point["warmup_steps"] == 400
    assert set(clipped) == {"learning_rate", "batch_size", "warmup_steps"} and filled == ["weight_decay"]
    assert point["weight_decay"] == pytest.approx(0.05) and coverage == 0.75


def test_suggest_is_deterministic_and_fills_with_halton():
    rng = random.Random(1)
    jobs = []
    for j in range(6):
        trials = [{"index": i, "params": {"learning_rate": 10 ** rng.uniform(-5, -1), "weight_decay": rng.random() / 10,
                                          "batch_size": rng.choice([32, 64, 128]), "warmup_steps": 100 * rng.randint(0, 20)},
                   "value": rng.random(), "status": "complete"} for i in range(30)]
        for t, p in zip(trials, alg.percentiles([t["value"] for t in trials], "minimize")):
            t["percentile"] = p
        jobs.append({"record_id": f"r{j}", "knowledge_base_id": "kb", "job_id": f"j{j}", "record_name": f"job {j}",
                     "search_space": SPACE, "objective": {"name": "val_loss", "direction": "minimize"},
                     "trials": trials, "n_complete": 30, "task": None})
    objective = {"name": "val_loss", "direction": "minimize"}
    first = alg.suggest(SPACE, objective, jobs, 5, 12, {"seed": 3})
    assert first == alg.suggest(SPACE, objective, jobs, 5, 12, {"seed": 3})
    warm = [p for p in first["points"] if p["source"].get("type") != "halton"]
    assert len(warm) == 5 and len(first["points"]) == 12 and len(first["source_jobs"]) == 5
    vectors = [alg.encode(p["params"], SPACE) for p in warm]
    assert all(alg.distance(a, b) >= alg.DEDUP_RADIUS for i, a in enumerate(vectors) for b in vectors[i + 1:])
    for point in first["points"]:
        assert SPACE["warmup_steps"]["low"] <= point["params"]["warmup_steps"] <= 2000
        assert point["params"]["warmup_steps"] % 100 == 0 and point["params"]["batch_size"] in (32, 64, 128)
    history = alg.suggest(SPACE, objective, jobs[:1], 2, 2, {"include_history": True})["history"]
    assert len(history) == 30


# --- end to end -------------------------------------------------------------------------------------------
@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        c.app.state.manager.services.records.sync_plugin_types and __import__("asyncio").run(
            _sync(c.app.state.manager.services.records))
        yield c


async def _sync(records):
    from agenticwf.kb.core.orm.base.session import dispose_async_engine

    try:
        await records.sync_plugin_types()
    finally:
        await dispose_async_engine()


def job(i, rng, dataset="cifar100", direction="minimize", offset=0.0):
    trials = []
    for n in range(20):
        lr = 10 ** rng.uniform(-5, -1)
        wd = rng.random() / 10
        value = (math.log10(lr) + 3 + offset) ** 2 + 5 * (wd - 0.03) ** 2 + rng.random() * 0.01
        trials.append({"number": n, "params": {"lr": lr, "weight_decay": wd, "batch_size": rng.choice([32, 64])},
                       "value": value, "status": "complete"})
    return {"job_id": f"job-{dataset}-{i}", "description": f"ResNet-50 fine-tuning on {dataset}",
            "objective": {"name": "val_loss", "direction": direction},
            "search_space": {"lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
                             "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
                             "batch_size": {"type": "categorical", "choices": [32, 64]}},
            "trials": trials, "metadata": {"dataset": dataset, "model": "resnet50", "framework": "optuna"}}


def test_suggest_end_to_end_and_access(client):
    alice = Caller(client, user("alice", RW, org="eic"))
    kb = alice.post("/kb/api/v1/knowledge-bases", json={
        "name": "HPO", "type": "records", "records_config": {
            "default_type": "hpo_job", "hpo": {"param_aliases": {"learning_rate": ["lr"]}}}}).json()["data"]
    rng = random.Random(0)
    report = alice.post(f"/kb/api/v1/knowledge-bases/{kb['id']}/records",
                        json={"records": [job(i, rng) for i in range(4)]}).json()["data"]
    assert report["created"] == 4, report
    drain(client)
    request = {"objective": {"name": "Validation_Loss", "direction": "minimize"},
               "description": "ResNet-50 on CIFAR-10",
               "search_space": {"learningRate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
                                "dropout": {"type": "float", "low": 0.0, "high": 0.5}},
               "n": 4, "n_total": 8, "options": {"min_similarity": 0.1, "min_coverage": 0.4}}
    # "validation_loss" is not an alias in this KB: no candidates, all space-filling
    cold = alice.post("/kb/api/v1/records/suggest", json=request).json()["data"]
    assert cold["source_jobs"] == [] and len(cold["points"]) == 8
    request["objective"]["name"] = "val_loss"
    data = alice.post("/kb/api/v1/records/suggest", json=request).json()["data"]
    assert len(data["source_jobs"]) == 4 and data["unmatched_params"] == ["dropout"]
    assert set(data["source_jobs"][0]["breakdown"]) == {"params", "ranges", "task"}
    warm = [p for p in data["points"] if "job_id" in p["source"]]
    assert len(warm) == 4 and all(p["filled_params"] == ["dropout"] for p in warm)
    assert all(1e-4 <= p["params"]["learning_rate"] <= 1e-2 for p in data["points"])
    assert data == alice.post("/kb/api/v1/record-types/hpo_job/ops/suggest", json=request).json()["data"]
    # items-level aggregate: best value per batch size
    agg = alice.post("/kb/api/v1/records/aggregate", json={
        "record_type": "hpo_job", "scope": "items", "group_by": ["params.batch_size"],
        "metrics": [{"fn": "min", "field": "value"}, {"fn": "count"}]}).json()["data"]
    assert sum(r["count"] for r in agg["rows"]) == 80
    # another user never sees alice's jobs as sources
    bob = Caller(client, user("bob", RW, org="atlas"))
    other = bob.post("/kb/api/v1/records/suggest", json=request).json()["data"]
    assert other["source_jobs"] == []
    assert bob.post("/kb/api/v1/records/suggest", json={**request, "knowledge_base_ids": [kb["id"]]}).status_code == 404


# --- the benchmark --------------------------------------------------------------------------------------------
def _objective(shift, scale, extra):
    def f(p):
        value = scale * ((p["x"] - shift[0]) ** 2 + (p["y"] - shift[1]) ** 2)
        if "z" in p:
            value += 0.2 * (p["z"] - extra) ** 2
        return value
    return f


def _space(with_z):
    space = {"x": {"type": "float", "low": -5.0, "high": 5.0}, "y": {"type": "float", "low": -5.0, "high": 5.0}}
    if with_z:
        space["z"] = {"type": "float", "low": -5.0, "high": 5.0}
    return space


def _past_jobs(rng, centre, count):
    jobs = []
    for j in range(count):
        shift = (centre[0] + rng.gauss(0, 0.3), centre[1] + rng.gauss(0, 0.3))
        f = _objective(shift, rng.uniform(0.5, 2.0), 0)
        space = _space(rng.random() < 0.5)
        trials = []
        for i in range(40):
            p = {k: rng.uniform(v["low"], v["high"]) for k, v in space.items()}
            trials.append({"index": i, "params": p, "value": f(p), "status": "complete"})
        for t, pct in zip(trials, alg.percentiles([t["value"] for t in trials], "minimize")):
            t["percentile"] = pct
        jobs.append({"record_id": f"r{j}", "knowledge_base_id": "kb", "job_id": f"j{j}", "search_space": space,
                     "objective": {"name": "f", "direction": "minimize"}, "trials": trials, "n_complete": 40,
                     "task": None})
    return jobs


def _regrets(f, space, initial, seed, checkpoints=(10, 20)):
    """Best-so-far value (the optimum is 0, so this is the regret) at each checkpoint."""
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    for point in initial:
        study.enqueue_trial(point)
    study.optimize(lambda t: f({k: t.suggest_float(k, v["low"], v["high"]) for k, v in space.items()}),
                   n_trials=max(checkpoints))
    values = [t.value for t in study.trials]
    return [min(values[:c]) for c in checkpoints]


def test_warm_start_benchmark():
    """Shifted, rescaled quadratics sharing x and y, with z added or removed at random. Related
    tasks: warm start lowers the median regret at trials 10 and 20. Unrelated tasks (the optimum
    moved far away): not meaningfully worse than cold start, thanks to the space-filling points."""
    pytest.importorskip("optuna")
    rng = random.Random(42)
    space = _space(True)
    objective = {"name": "f", "direction": "minimize"}
    related = {"warm": [], "cold": []}
    unrelated = {"warm": [], "cold": []}
    for task in range(12):
        centre = (rng.uniform(-3, 3), rng.uniform(-3, 3))
        jobs = _past_jobs(rng, centre, 6)
        warm = [p["params"] for p in alg.suggest(space, objective, jobs, 5, 10, {"seed": task})["points"]]
        cold = [p["params"] for p in alg.suggest(space, objective, [], 0, 10, {"seed": task})["points"]]
        near = _objective((centre[0] + rng.gauss(0, 0.3), centre[1] + rng.gauss(0, 0.3)), 1.0, 0)
        far = _objective((-centre[0], -centre[1]), 1.0, 2.0)
        for seed in range(3):
            related["warm"].append(_regrets(near, space, warm, seed))
            related["cold"].append(_regrets(near, space, cold, seed))
            unrelated["warm"].append(_regrets(far, space, warm, seed))
            unrelated["cold"].append(_regrets(far, space, cold, seed))

    def median_at(runs, i):
        return statistics.median(r[i] for r in runs)

    for i in (0, 1):                                            # trial 10, trial 20
        assert median_at(related["warm"], i) < median_at(related["cold"], i)
    assert median_at(unrelated["warm"], 1) <= median_at(unrelated["cold"], 1) + 0.5


def test_python_client_warm_start_round_trip(client):
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from conftest import signed
    from agenticwf.kb.client import KBClient
    from agenticwf.kb.client.hpo import upload_study, warm_start

    ctx = user("alice", RW, org="eic")
    kb = KBClient("http://testserver/kb", http=client,
                  headers=lambda method, url: signed(ctx, method, url.split("testserver", 1)[1].split("?", 1)[0]))
    kb_id = kb.request("POST", "/knowledge-bases", json={"name": "HPO", "type": "records"})["id"]
    space = {"x": {"type": "float", "low": -5.0, "high": 5.0}, "y": {"type": "int", "low": 0, "high": 10}}
    past = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    past.optimize(lambda t: (t.suggest_float("x", -5, 5) - 1) ** 2 + (t.suggest_int("y", 0, 10) - 3) ** 2, n_trials=25)
    report = upload_study(kb, past, kb_id, "past-1", search_space=space, objective_name="loss",
                          description="a quadratic")
    assert report["created"] == 1
    new = optuna.create_study(direction="minimize")
    data = warm_start(kb, new, space, [kb_id], objective_name="loss", n=3, n_total=5, include_history=True)
    assert len(data["points"]) == 5 and len(data["history"]) == 25
    states = [t.state.name for t in new.trials]
    assert states.count("COMPLETE") == 25 and states.count("WAITING") == 5     # history added, points enqueued
    new.optimize(lambda t: (t.suggest_float("x", -5, 5) - 1) ** 2 + (t.suggest_int("y", 0, 10) - 3) ** 2, n_trials=5)
    assert new.best_value <= past.best_value + 1e-9
