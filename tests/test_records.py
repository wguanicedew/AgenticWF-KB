"""Structured records (docs/structured-records.md): types, ingest, query, aggregate, similar,
statistics, evolution, access, and the compiled-SQL vs reference-evaluator property test."""

import io
import json
import random

import pytest

from conftest import Caller, drain, make_client, sql_rows, user
from agenticwf.kb.runtime.errors import Invalid
from agenticwf.kb.runtime.records import query as rq
from agenticwf.kb.runtime.records import similarity as sim
from agenticwf.kb.runtime.records.service import prerank_sql, text_words, type_text
from agenticwf.kb.runtime.records.schema import TypeDef, classify_change, infer

RW = ("app:access", "knowledge:read", "knowledge:write")

EXPERIMENT_RUN = {
    "name": "experiment_run", "title": "ML experiment run", "visibility": "private",
    "schema": {"type": "object", "required": ["run_id", "model", "dataset", "metrics"], "properties": {
        "run_id": {"type": "string"}, "run_name": {"type": "string"}, "model": {"type": "string"},
        "dataset": {"type": "string"}, "finished_at": {"type": "string", "format": "date-time"},
        "notes": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}},
        "params": {"type": "object", "properties": {"lr": {"type": "number", "x-log": True},
                                                    "batch_size": {"type": "integer"},
                                                    "optimizer": {"type": "string"}}},
        "metrics": {"type": "object", "properties": {"accuracy": {"type": "number"},
                                                     "train_hours": {"type": "number"}}},
        "epochs": {"type": "array", "items": {"type": "object", "properties": {
            "epoch": {"type": "integer"}, "loss": {"type": "number"}}}}}},
    "roles": {"id": "run_id", "name": "run_name", "text": ["notes"], "time": "finished_at",
              "facets": ["model", "dataset", "params.optimizer"],
              "measures": {"metrics.accuracy": {"direction": "maximize"},
                           "metrics.train_hours": {"unit": "h", "direction": "minimize"}},
              "features": ["params.lr", "params.batch_size", "params.optimizer"],
              "items": {"path": "epochs", "measures": {"loss": {"direction": "minimize"}}, "features": ["epoch"]}},
    "similarity": {"weights": {"text": 0.3, "facets": 0.3, "features": 0.4}},
    "aliases": {"params.lr": ["params.learning_rate", "params.eta"]},
}

CSV = """run,model,dataset,opt,lr,bs,acc,hours,when,notes
r1,resnet50,cifar100,adamw,0.001,64,0.81,3.5,2026-01-10T10:00:00Z,label noise in the tail classes
r2,resnet50,cifar100,sgd,0.1,128,0.78,2.0,2026-01-20T10:00:00Z,plain baseline
r3,vit-b16,cifar100,adamw,0.0003,64,0.86,6.0,2026-02-05T10:00:00Z,strong augmentation
r4,vit-b16,cifar10,adamw,0.0003,64,0.97,4.0,2026-02-15T10:00:00Z,easy dataset
r5,resnet50,cifar100,adamw,0.002,32,0.83,4.5,2026-03-01T10:00:00Z,label noise again
"""
COLUMNS = {"run": "run_id", "opt": "params.optimizer", "lr": "params.lr", "bs": "params.batch_size",
           "acc": "metrics.accuracy", "hours": "metrics.train_hours", "when": "finished_at"}


def typedef():
    return TypeDef(name="experiment_run", qualified="alice:experiment_run", version=1,
                   schema=EXPERIMENT_RUN["schema"], roles=EXPERIMENT_RUN["roles"], aliases=EXPERIMENT_RUN["aliases"])


# --- units ---------------------------------------------------------------------------------------------
def test_roles_aliases_and_extraction():
    td = typedef()
    td.check_roles()
    record = td.canonicalise({"run_id": "x", "model": "m", "dataset": "d", "metrics": {"accuracy": 0.5},
                              "params": {"learning_rate": 0.01, "optimizer": "sgd"},
                              "finished_at": "2026-01-01T00:00:00Z", "epochs": [{"epoch": 1, "loss": 2.0}]})
    assert record["params"] == {"lr": 0.01, "optimizer": "sgd"}
    assert td.validate(record) == []
    assert td.validate({**record, "unknown": 1}) == ["unknown: not in the schema"]
    ex = td.extract(record)
    assert ex["external_id"] == "x" and ex["name"] == "x" and ex["facets"] == {"model": "m", "dataset": "d",
                                                                               "params.optimizer": "sgd"}
    assert ex["measures"] == {"metrics.accuracy": 0.5} and ex["items"][0]["measures"] == {"loss": 2.0}
    assert td.field("params.eta").path == "params.lr"
    with pytest.raises(Invalid):
        td.field("params.momentum")
    bad = dict(EXPERIMENT_RUN["roles"], measures={"model": {}})
    with pytest.raises(Invalid):
        TypeDef(name="x", qualified="a:x", version=1, schema=EXPERIMENT_RUN["schema"], roles=bad).check_roles()


def test_query_compilation_rejects_and_binds():
    td = typedef()
    sql, _, params, _ = rq.compile_query(td, {"filter": {"and": [
        {"field": "dataset", "op": "eq", "value": "cifar100"},
        {"field": "params.eta", "op": "lt", "value": 0.01},
        {"scope": "items", "field": "loss", "op": "lt", "value": 0.2}]}, "sort": [{"field": "metrics.accuracy",
                                                                               "dir": "desc"}]}, 1000)
    assert "cifar100" not in sql and "EXISTS (SELECT 1 FROM kb_record_items" in sql
    assert ["params", "lr"] in params.values()
    for body in ({"filter": {"field": "data; DROP TABLE kb_records", "op": "eq", "value": 1}},
                 {"filter": {"field": "model", "op": "regex", "value": "x"}},
                 {"filter": {"field": "model", "op": "lt", "value": True}, "limit": 5000},
                 {"sort": [{"field": "model", "dir": "sideways"}]}):
        with pytest.raises(Invalid):
            rq.compile_query(td, body, 1000)
    agg_sql, _, groups, metrics = rq.compile_aggregate(td, {
        "group_by": ["model", {"field": "finished_at", "bucket": "month"}],
        "metrics": [{"fn": "count"}, {"fn": "best", "field": "metrics.accuracy"},
                    {"fn": "p90", "field": "metrics.train_hours"}],
        "having": {"field": "count", "op": "gte", "value": 1}}, 1000)
    assert groups == ["model", "finished_at:month"] and metrics == ["count", "best(metrics.accuracy)",
                                                                    "p90(metrics.train_hours)"]
    assert "max(" in agg_sql and "percentile_cont(0.9)" in agg_sql and "date_trunc('month'" in agg_sql
    with pytest.raises(Invalid):
        rq.compile_aggregate(td, {"metrics": [{"fn": "best", "field": "params.lr"}]}, 1000)     # no direction
    with pytest.raises(Invalid):
        rq.compile_aggregate(td, {"group_by": [{"field": "finished_at", "bucket": "fortnight"}]}, 1000)


def test_similarity_components():
    assert sim.feature_similarity(0.1, 0.1, 0.0, 1.0) == 1.0
    assert sim.feature_similarity(0.001, 0.01, 0.0001, 0.1, log=True) == pytest.approx(1 - 1 / 3)
    assert sim.feature_similarity("adam", "sgd") == 0.0
    assert sim.feature_similarity(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert sim.features_similarity({"x": 1, "y": 2}, {"x": 1}, {"x": (0, 2)}, set()) == 1.0     # missing ≠ penalty
    assert sim.combine({"text": None, "facets": 1.0, "features": 0.5}) == pytest.approx((0.3 + 0.2) / 0.7)



def test_similar_prerank_and_type_text():
    expr, params = prerank_sql({"model": "resnet50", "flag": True}, {"params.lr": 0.001, "params.optimizer": "adamw",
                                                                     "tags": ["a"]},
                               {"params.lr": (0.0001, 0.1)}, {"params.lr"})
    assert expr.count("CASE WHEN r.facets") == 2 and "ln(" in expr and "r.features ->> " in expr
    values = set(map(str, params.values()))
    assert {"model", "resnet50", "true", "params.optimizer", "adamw"} <= values and "tags" not in values
    assert prerank_sql({}, {}, {}, set()) == ("0", {})
    a = {"record_type": "t", "score": 0.9, "facets": {"m": "x"}, "features": {"lr": 0.001}}
    twin = {**a, "score": 0.85}
    other = {"record_type": "t", "score": 0.6, "facets": {"m": "y"}, "features": {"lr": 0.1}}
    ranges = {"lr": (0.001, 0.1)}
    assert sim.record_similarity(a, twin, ranges, set()) == 1.0
    assert sim.record_similarity(a, {**twin, "record_type": "u"}, ranges, set()) == 0.0
    alike = lambda x, y: sim.record_similarity(x, y, ranges, set())     # noqa: E731
    assert sim.mmr([a, twin, other], 2, alike, 1.0) == [a, twin]            # λ = 1: plain ranking
    assert sim.mmr([a, twin, other], 2, alike, 0.3) == [a, other]           # the twin is redundant
    assert sim.cosine([1, 0], [1, 0]) == 1.0 and sim.cosine([1, 0], [0, 1]) == 0.0 and sim.cosine([0], [0]) == 0.0

    class Row:
        name, title, description = "experiment_run", "ML experiment run", "One training run"
    words = text_words(type_text(Row, typedef()))
    assert {"experiment", "run", "training", "lr", "optimizer", "accuracy"} <= words

def test_inference_rules():
    samples = [{"run_id": f"r{i}", "model": ["resnet", "vit"][i % 2], "accuracy": 0.8 + i / 100,
                "val_loss": 1.0 - i / 100, "finished_at": f"2026-01-{i + 1:02d}T00:00:00Z",
                "notes": "a long description of what happened in this training run, " * 3,
                "params": {"lr": 0.001 * (i + 1)}, "trials": [{"loss": 0.5, "step": 1}]} for i in range(20)]
    proposal = infer(samples, "runs")
    roles = proposal["roles"]
    assert roles["id"] == "run_id" and roles["time"] == "finished_at" and roles["text"] == ["notes"]
    assert "model" in roles["facets"] and roles["measures"]["accuracy"]["direction"] == "maximize"
    assert roles["measures"]["val_loss"]["direction"] == "minimize" and "params.lr" in roles["features"]
    assert roles["items"]["path"] == "trials" and "loss" in roles["items"]["measures"]
    TypeDef(name="runs", qualified="a:runs", version=1, schema=proposal["schema"], roles=roles).check_roles()


def test_change_classification():
    old = typedef()
    additive = json.loads(json.dumps(EXPERIMENT_RUN["schema"]))
    additive["properties"]["seed"] = {"type": "integer"}
    assert classify_change(old, TypeDef(name="e", qualified="a:e", version=1, schema=additive,
                                        roles=EXPERIMENT_RUN["roles"])) == []
    breaking = json.loads(json.dumps(EXPERIMENT_RUN["schema"]))
    breaking["properties"]["notes"] = {"type": "integer"}
    breaking["required"].append("run_name")
    reasons = classify_change(old, TypeDef(name="e", qualified="a:e", version=1, schema=breaking,
                                           roles=EXPERIMENT_RUN["roles"]))
    assert any("notes" in r for r in reasons) and any("run_name" in r for r in reasons)


# --- end to end -------------------------------------------------------------------------------------------
@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


@pytest.fixture
def alice(client):
    return Caller(client, user("alice", RW, org="eic"))


def ok(response, status=200):
    assert response.status_code == status, response.text
    return response.json()["data"]


def setup_runs(client, alice):
    ok(alice.post("/kb/api/v1/record-types", json=EXPERIMENT_RUN), 201)
    kb = ok(alice.post("/kb/api/v1/knowledge-bases", json={
        "name": "Runs", "type": "records",
        "records_config": {"allowed_types": ["experiment_run"], "default_type": "experiment_run"}}), 201)
    assert kb["records_config"]["default_type"] == "alice:experiment_run"
    upload = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb['id']}/knowledge/file",
                           files={"file": ("runs.csv", io.BytesIO(CSV.encode()), "text/csv")},
                           data={"column_map": json.dumps(COLUMNS)}), 201)
    drain(client)
    item = ok(alice.get(f"/kb/api/v1/knowledge/{upload['id']}"))
    assert item["parse_status"] == "completed", item
    assert item["metadata"]["records_report"]["created"] == 5, item["metadata"]
    return kb["id"]


def test_experiment_runs_end_to_end(client, alice):
    kb_id = setup_runs(client, alice)
    listed = alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/records?facet.model=vit-b16").json()
    assert listed["total"] == 2
    q = ok(alice.post("/kb/api/v1/records/query", json={
        "knowledge_base_ids": [kb_id], "record_type": "experiment_run",
        "filter": {"and": [{"field": "dataset", "op": "eq", "value": "cifar100"},
                           {"field": "metrics.accuracy", "op": "gte", "value": 0.8},
                           {"field": "finished_at", "op": "between", "value": ["2026-01-01", "2026-02-28"]}]},
        "sort": [{"field": "metrics.accuracy", "dir": "desc"}], "fields": ["model", "metrics.accuracy"]}))
    assert [r["external_id"] for r in q["records"]] == ["r3", "r1"] and q["total"] == 2
    assert q["records"][0]["fields"] == {"model": "vit-b16", "metrics.accuracy": 0.86}
    agg = ok(alice.post("/kb/api/v1/records/aggregate", json={
        "record_type": "experiment_run", "filter": {"field": "dataset", "op": "eq", "value": "cifar100"},
        "group_by": ["params.optimizer"], "metrics": [{"fn": "count"}, {"fn": "best", "field": "metrics.accuracy"},
                                                       {"fn": "avg", "field": "metrics.train_hours"}],
        "sort": [{"field": "best(metrics.accuracy)", "dir": "desc"}]}))
    assert agg["rows"][0] == {"params.optimizer": "adamw", "count": 3, "best(metrics.accuracy)": 0.86,
                              "avg(metrics.train_hours)": pytest.approx(14 / 3)}
    months = ok(alice.post("/kb/api/v1/records/aggregate", json={
        "record_type": "experiment_run", "group_by": [{"field": "finished_at", "bucket": "month"}],
        "metrics": [{"fn": "count"}], "having": {"field": "count", "op": "gte", "value": 2}}))
    assert [r["count"] for r in months["rows"]] == [2, 2]
    r1 = [r for r in ok(alice.get(f"/kb/api/v1/knowledge-bases/{kb_id}/records"))  # list
          if r["external_id"] == "r1"][0]
    similar = ok(alice.post("/kb/api/v1/records/similar", json={"record_id": r1["id"], "limit": 2}))
    assert similar["records"][0]["name"] == "r5"           # same model, dataset, optimizer, close lr
    assert set(similar["records"][0]["breakdown"]) == {"text", "facets", "features"}
    probe = ok(alice.post("/kb/api/v1/records/similar", json={
        "record_type": "experiment_run", "probe": {"model": "vit-b16", "dataset": "cifar10",
                                                   "params": {"optimizer": "adamw", "learning_rate": 0.0003}}}))
    assert probe["records"][0]["external_id"] == "r4"
    stats = {s["field"]: s for s in ok(alice.get("/kb/api/v1/record-types/experiment_run/stats"))}
    assert stats["metrics.accuracy"]["min"] == 0.78 and stats["metrics.accuracy"]["max"] == 0.97
    assert {"value": "resnet50", "count": 3} in stats["model"]["top_values"]
    # text roles reach hybrid search after the index task
    hits = ok(alice.post("/kb/api/v1/knowledge-search", json={"query": "label noise", "knowledge_base_ids": [kb_id]}))
    assert hits and hits[0]["chunk_type"] == "record" and hits[0]["record_type"] == "alice:experiment_run"
    text = ok(alice.post("/kb/api/v1/records/query", json={
        "record_type": "experiment_run", "filter": {"field": "notes", "op": "text", "value": "label noise"}}))
    assert {r["external_id"] for r in text["records"]} == {"r1", "r5"}



INCIDENT = {
    "name": "incident", "title": "Operations incident", "visibility": "private",
    "description": "An outage or degradation of a production service, with severity and duration",
    "schema": {"type": "object", "required": ["key", "service"], "properties": {
        "key": {"type": "string"}, "title": {"type": "string"}, "service": {"type": "string"},
        "severity": {"type": "string"}, "summary": {"type": "string"}, "minutes": {"type": "number"}}},
    "roles": {"id": "key", "name": "title", "text": ["summary"], "facets": ["service", "severity"],
              "measures": {"minutes": {"unit": "min", "direction": "minimize"}}, "features": ["minutes"]},
}


def test_similar_in_stages(client, alice):
    runs = setup_runs(client, alice)
    ok(alice.post("/kb/api/v1/record-types", json=INCIDENT), 201)
    ops = ok(alice.post("/kb/api/v1/knowledge-bases", json={
        "name": "Ops", "type": "records", "records_config": {"default_type": "incident"}}), 201)["id"]
    ok(alice.post(f"/kb/api/v1/knowledge-bases/{ops}/records", json={"records": [
        {"key": "i1", "title": "db outage", "service": "postgres", "severity": "high",
         "summary": "primary database disk full, writes failed", "minutes": 42},
        {"key": "i2", "title": "slow api", "service": "gateway", "severity": "low",
         "summary": "latency spike after deploy", "minutes": 15}]}))
    drain(client)

    # narrow: only records matching the text, ranked with the probe's parameters
    narrow = ok(alice.post("/kb/api/v1/records/similar", json={
        "record_type": "experiment_run", "query": "label noise", "probe": {"params": {"learning_rate": 0.001}}}))
    assert narrow["mode"] == "narrow" and narrow["anchor"] is None
    assert [r["external_id"] for r in narrow["records"]] == ["r1", "r5"]       # r1 has exactly lr 0.001
    assert all(r["text_rank"] and r["breakdown"]["text"] is not None for r in narrow["records"])

    # expand (the default for a query alone): the best text match is the anchor, then parameters
    expand = ok(alice.post("/kb/api/v1/records/similar", json={
        "query": "label noise in the tail classes", "knowledge_base_ids": [runs]}))
    assert expand["mode"] == "expand" and expand["anchor"]["external_id"] == "r1"
    ids = [r["external_id"] for r in expand["records"]]
    assert "r1" not in ids and ids[0] == "r5" and set(ids) == {"r2", "r3", "r4", "r5"}
    assert all(r["breakdown"]["text"] is None for r in expand["records"])
    assert expand["records"][-1]["external_id"] == "r2"                            # sgd, lr 0.1: furthest
    assert expand["diversified"] is False and "features" not in expand["records"][0]
    spread = ok(alice.post("/kb/api/v1/records/similar", json={
        "query": "label noise in the tail classes", "limit": 3, "diversify": True, "mmr_lambda": 0.3}))
    assert [r["external_id"] for r in spread["records"]] == ["r5", "r2", "r4"]    # r3 is too close to r5
    assert spread["diversified"] is True
    assert alice.post("/kb/api/v1/records/similar", json={
        "query": "label noise", "diversify": True, "mmr_lambda": 2}).status_code == 400

    # stage 0: no record_type, the query picks the type
    routed = ok(alice.post("/kb/api/v1/records/similar", json={"query": "database outage"}))
    assert routed["record_type"] == "alice:incident" and routed["anchor"]["external_id"] == "i1"
    types = {t["record_type"]: t for t in routed["types"]}
    assert types["alice:incident"]["chosen"] and not types["alice:experiment_run"]["chosen"]
    assert types["alice:incident"]["score"] > types["alice:experiment_run"]["score"]
    assert {r["record_type"] for r in routed["records"]} == {"alice:incident"}

    assert alice.post("/kb/api/v1/records/similar", json={"limit": 3}).status_code == 400
    assert alice.post("/kb/api/v1/records/similar", json={"query": "x", "mode": "sideways"}).status_code == 400

def test_upsert_versions_items_and_freshness(client, alice):
    ok(alice.post("/kb/api/v1/record-types", json=EXPERIMENT_RUN), 201)
    kb_id = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "R", "type": "records"}), 201)["id"]
    run = {"run_id": "e1", "model": "m", "dataset": "d", "metrics": {"accuracy": 0.5},
           "epochs": [{"epoch": 1, "loss": 0.9}, {"epoch": 2, "loss": 0.15}]}
    first = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/records",
                          json={"record_type": "experiment_run", "records": [run, {"run_id": "bad"}]}))
    assert first["created"] == 1 and first["invalid"] == 1 and "model" in " ".join(first["outcomes"][1]["errors"])
    again = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/records", json={"record_type": "experiment_run",
                                                                               "records": [run]}))
    assert again["unchanged"] == 1
    atomic = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/records", json={
        "record_type": "experiment_run", "atomic": True, "records": [{**run, "run_id": "e2"}, {"run_id": "bad"}]}))
    assert atomic["written"] is False
    changed = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/records", json={
        "record_type": "experiment_run", "records": [{**run, "metrics": {"accuracy": 0.6}, "epochs": [{"epoch": 1,
                                                                                                  "loss": 0.3}]}]}))
    assert changed["updated"] == 1
    record = ok(alice.get(f"/kb/api/v1/records/{changed['outcomes'][0]['id']}"))
    assert record["version"] == 2 and record["versions"] == [1] and len(record["items"]) == 1
    old = ok(alice.get(f"/kb/api/v1/records/{record['id']}?version=1"))
    assert old["data"]["metrics"]["accuracy"] == 0.5
    items = ok(alice.post("/kb/api/v1/records/query", json={
        "record_type": "experiment_run", "filter": {"scope": "items", "field": "loss", "op": "lt", "value": 0.2}}))
    assert items["total"] == 0                                    # the items were replaced
    ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/records", json={
        "record_type": "experiment_run", "records": [{**run, "run_id": "old", "_deprecated": True},
                                                     {**run, "run_id": "future", "_valid_from": "2099-01-01"}]}))
    visible = ok(alice.post("/kb/api/v1/records/query", json={"record_type": "experiment_run"}))
    assert {r["external_id"] for r in visible["records"]} == {"e1"}
    alice.delete(f"/kb/api/v1/records/{record['id']}")
    assert alice.get(f"/kb/api/v1/records/{record['id']}").status_code == 404


def test_schema_evolution_and_migration(client, alice):
    kb_id = setup_runs(client, alice)
    additive = json.loads(json.dumps(EXPERIMENT_RUN))
    additive["schema"]["properties"]["seed"] = {"type": "integer"}
    updated = ok(alice.put("/kb/api/v1/record-types/experiment_run", json={"schema": additive["schema"]}))
    assert updated["version"] == 1 and updated["breaking"] == []
    breaking = json.loads(json.dumps(additive["schema"]))
    breaking["properties"]["summary"] = breaking["properties"].pop("notes")
    roles = dict(EXPERIMENT_RUN["roles"], text=["summary"])
    conflict = alice.put("/kb/api/v1/record-types/experiment_run", json={"schema": breaking, "roles": roles})
    assert conflict.status_code == 409 and "notes removed" in conflict.text
    v2 = ok(alice.put("/kb/api/v1/record-types/experiment_run",
                      json={"schema": breaking, "roles": roles, "new_version": True}))
    assert v2["version"] == 2
    # old records stay queryable through roles
    assert ok(alice.post("/kb/api/v1/records/query", json={"record_type": "experiment_run"}))["total"] == 5
    task = ok(alice.post("/kb/api/v1/record-types/experiment_run/migrate", json={"mapping": {"notes": "summary"}}))
    assert kb_id in task["knowledge_base_ids"]
    drain(client)
    versions = sql_rows("SELECT DISTINCT type_version FROM kb_records WHERE knowledge_base_id = :k AND deleted_at IS NULL",
                        k=kb_id)
    assert versions == [(2,)]
    text = ok(alice.post("/kb/api/v1/records/query", json={"record_type": "experiment_run",
                                                            "filter": {"field": "summary", "op": "exists"}}))
    assert text["total"] == 5
    ok(alice.delete("/kb/api/v1/record-types/experiment_run"))
    rejected = ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/records", json={"run_id": "z"}))
    assert rejected["invalid"] == 1 and "deprecated" in rejected["outcomes"][0]["errors"][0]


def test_access(client, alice):
    kb_id = setup_runs(client, alice)
    bob = Caller(client, user("bob", RW, org="atlas"))
    assert bob.get("/kb/api/v1/record-types/alice:experiment_run").status_code == 404
    assert bob.get(f"/kb/api/v1/knowledge-bases/{kb_id}/records").status_code == 404
    assert bob.post("/kb/api/v1/records/query", json={"knowledge_base_ids": [kb_id],
                                                      "record_type": "alice:experiment_run"}).status_code == 404
    record_id = sql_rows("SELECT id FROM kb_records LIMIT 1")[0][0]
    assert bob.get(f"/kb/api/v1/records/{record_id}").status_code == 404
    # sharing the KB makes the type readable through it, and its stats
    alice.put(f"/kb/api/v1/knowledge-bases/{kb_id}", json={"visibility": "groups", "groups": ["org:eic"]})
    carol = Caller(client, user("carol", RW, org="eic"))
    assert ok(carol.get("/kb/api/v1/record-types/alice:experiment_run"))["name"] == "alice:experiment_run"
    assert ok(carol.post("/kb/api/v1/records/query", json={"record_type": "alice:experiment_run"}))["total"] == 5
    assert ok(bob.get("/kb/api/v1/record-types")) == [t for t in ok(bob.get("/kb/api/v1/record-types"))
                                                      if t["owner_subject"] == "system"]
    assert ok(carol.get("/kb/api/v1/record-types/alice:experiment_run/stats"))
    assert bob.get("/kb/api/v1/record-types/alice:experiment_run/stats").status_code == 404


def test_infer_endpoint(client, alice):
    proposal = ok(alice.post("/kb/api/v1/record-types/infer", files={"file": ("runs.csv", io.BytesIO(CSV.encode()),
                                                                              "text/csv")}, data={"name": "runs"}))
    assert proposal["roles"]["id"] == "run" and proposal["suggested_by"] == "rules"


# --- property test: compiled SQL = Python reference -----------------------------------------------------
def random_filter(rng, depth=0):
    if depth < 2 and rng.random() < 0.35:
        key = rng.choice(["and", "or"])
        return {key: [random_filter(rng, depth + 1) for _ in range(rng.randint(2, 3))]}
    if depth < 2 and rng.random() < 0.1:
        return {"not": random_filter(rng, depth + 1)}
    if rng.random() < 0.15:
        return {"scope": "items", "field": "loss", "op": rng.choice(["lt", "gte"]), "value": round(rng.random(), 2)}
    choice = rng.choice(["model", "dataset", "metrics.accuracy", "params.lr", "params.batch_size", "finished_at",
                         "tags", "notes"])
    if choice in ("model", "dataset"):
        op = rng.choice(["eq", "ne", "in", "nin", "lt", "contains", "exists"])
        pool = ["a", "b", "c", "resnet", "vit"]
        value = rng.sample(pool, 2) if op in ("in", "nin") else rng.choice(pool)
    elif choice == "finished_at":
        op = rng.choice(["gte", "lt", "between", "exists"])
        day = lambda: f"2026-0{rng.randint(1, 3)}-1{rng.randint(0, 9)}T00:00:00Z"
        value = sorted([day(), day()]) if op == "between" else day()
    elif choice == "tags":
        op, value = "contains", rng.choice(["x", "y"])
    elif choice == "notes":
        op, value = rng.choice(["contains", "exists"]), rng.choice(["noise", "base"])
    else:
        op = rng.choice(["eq", "ne", "lt", "lte", "gt", "gte", "between", "in", "near", "exists"])
        number = lambda: round(rng.random(), 2) if choice != "params.batch_size" else rng.choice([32, 64, 128])
        if op == "between":
            value = sorted([number(), number()])
        elif op == "in":
            value = [number(), number()]
        elif op == "near":
            value = [number(), 0.2]
        else:
            value = number()
    node = {"field": choice, "op": op}
    if op != "exists":
        node["value"] = value
    elif rng.random() < 0.3:
        node["value"] = False
    return node


def random_record(rng, i):
    record = {"run_id": f"p{i}", "model": rng.choice(["a", "b", "c", "resnet", "vit"]),
              "dataset": rng.choice(["a", "b", "c"]), "metrics": {"accuracy": round(rng.random(), 2)}}
    if rng.random() < 0.8:
        record["params"] = {"lr": round(rng.random(), 2), "batch_size": rng.choice([32, 64, 128])}
    if rng.random() < 0.7:
        record["finished_at"] = f"2026-0{rng.randint(1, 3)}-1{rng.randint(0, 9)}T00:00:00Z"
    if rng.random() < 0.6:
        record["tags"] = rng.sample(["x", "y", "z"], rng.randint(0, 2))
    if rng.random() < 0.5:
        record["notes"] = rng.choice(["label noise", "baseline run", "Noise"])
    record["epochs"] = [{"epoch": e, "loss": round(rng.random(), 2)} for e in range(rng.randint(0, 3))]
    return record


def test_compiled_sql_matches_reference(client, alice):
    ok(alice.post("/kb/api/v1/record-types", json=EXPERIMENT_RUN), 201)
    kb_id = ok(alice.post("/kb/api/v1/knowledge-bases", json={"name": "P", "type": "records"}), 201)["id"]
    rng = random.Random(7)
    records = [random_record(rng, i) for i in range(60)]
    assert ok(alice.post(f"/kb/api/v1/knowledge-bases/{kb_id}/records",
                         json={"record_type": "experiment_run", "records": records}))["created"] == 60
    td = typedef()
    from agenticwf.kb.runtime.records.schema import parse_time
    for n in range(120):
        node = random_filter(rng)
        expected = {r["run_id"] for r in records if rq.evaluate(
            td, node, td.canonicalise(r), parse_time(r.get("finished_at")), r.get("epochs"))}
        got = ok(alice.post("/kb/api/v1/records/query", json={"record_type": "experiment_run", "filter": node,
                                                               "limit": 1000}))
        assert {r["external_id"] for r in got["records"]} == expected, (n, node)


# --- agentic search over records -------------------------------------------------------------------------
def test_agentic_answers_from_an_aggregate_with_record_citations(client, alice):
    from test_agentic import ScriptedChat, call, chat_model

    kb_id = setup_runs(client, alice)
    model = chat_model(alice)
    fake = ScriptedChat([
        {"tool_calls": [call("describe_record_types")]},
        {"tool_calls": [call("aggregate_records", record_type="experiment_run",
                             filter={"field": "dataset", "op": "eq", "value": "cifar100"},
                             group_by=["params.optimizer"], metrics=[{"fn": "best", "field": "metrics.accuracy"}],
                             sort=[{"field": "best(metrics.accuracy)", "dir": "desc"}]),
                        call("query_records", record_type="experiment_run",
                             filter={"field": "dataset", "op": "eq", "value": "cifar100"},
                             sort=[{"field": "metrics.accuracy", "dir": "desc"}], limit=1)]},
        {"content": 'AdamW: best accuracy 0.86 on cifar100 <ref id="r1"/>.'}])

    async def chat(ref):
        return fake

    client.app.state.manager.services.models.chat = chat
    data = ok(alice.post("/kb/api/v1/agentic-search", json={"query": "which optimizer gives the best accuracy on "
                                                                     "cifar100?", "knowledge_base_ids": [kb_id],
                                                            "model_id": model}))
    tools = [t["function"]["name"] for t in fake.calls[0]["tools"]]
    assert "aggregate_records" in tools and "grep_knowledge" not in tools        # records-only scope
    described = [m["content"] for m in fake.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "measure metrics.accuracy maximize: range 0.78 .. 0.97" in described
    table, rows = [m["content"] for m in fake.calls[2]["messages"] if m["role"] == "tool"][1:]
    assert "| adamw | 0.86 |" in table and "Copy numbers exactly" in table
    assert '<record id="r1" name="r3"' in rows
    assert '<record id="' in data["answer"] and 'type="alice:experiment_run" name="r3"' in data["answer"]
    assert data["evidence"][0]["kind"] == "record" and data["evidence"][0]["cited"] is True


def test_records_tools_forgive_small_model_mistakes(client, alice):
    """A stringified filter in Python syntax, an invented type name with one type in scope, and a
    bare field name all resolve; an ambiguous or unknown field still errors."""
    import asyncio

    from agenticwf.kb.runtime.agentic.handles import Handles
    from agenticwf.kb.runtime.agentic.tools import Scope, ToolContext, ToolError
    from agenticwf.kb.runtime.agentic.tools.records import aggregate_records, find_similar_records, query_records
    from agenticwf.kb.core.orm.base.session import dispose_async_engine

    kb_id = setup_runs(client, alice)
    services = client.app.state.manager.services

    async def run(tool, args):
        try:
            from agenticwf.kb.core.orm.base.session import async_session_scope
            from agenticwf.kb.core.orm.models.kb import KnowledgeBase

            async with async_session_scope(commit=False) as session:
                kb = await session.get(KnowledgeBase, kb_id)
                session.expunge(kb)
            tc = ToolContext(alice.ctx, services, Scope([kb_id], {kb_id: kb}), Handles(), lambda f: None)
            return await tool(tc, args)
        finally:
            await dispose_async_engine()

    out = asyncio.run(run(query_records, {"record_type": "best_optimizer", "fields": ["accuracy", "optimizer"],
                                          "filter": "{'field': 'dataset', 'op': 'eq', 'value': 'cifar100'}",
                                          "sort": '[{"field": "accuracy", "dir": "desc"}]', "limit": 1}))
    assert 'total="4"' in out and "metrics.accuracy=0.86" in out and "params.optimizer=adamw" in out
    table = asyncio.run(run(aggregate_records, {"record_type": "experiment_run", "group_by": "optimizer",
                                                "metrics": [{"fn": "best", "field": "accuracy"}]}))
    assert "| adamw | 0.97 |" in table
    similar = asyncio.run(run(find_similar_records, {"query": "label noise in the tail classes", "limit": 2}))
    assert 'mode="expand"' in similar and '<anchor id="r1" name="r1"' in similar
    assert similar.count("<record ") == 2 and 'name="r5"' in similar
    with pytest.raises(ToolError):
        asyncio.run(run(find_similar_records, {"probe": {"model": "resnet50"}}))      # a probe needs its type
    with pytest.raises(ToolError):
        asyncio.run(run(query_records, {"record_type": "experiment_run", "filter": {"field": "nonsense", "op": "eq",
                                                                                     "value": 1}}))
