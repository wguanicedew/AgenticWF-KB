"""Retrieval eval (roadmap phase 1, item 12): a small seed corpus and questions; the expected
document must be in the top 5. With the `hash` embedder vector search matches words only, so this
checks the pipeline (BM25 + fusion + filters) rather than embedding quality; point
[models] default_embedding at a real model to measure that."""

import pytest

from conftest import Caller, drain, make_client, user

CORPUS = {
    "pilot.md": "# Pilot\n\nRestart the pilot after three stage-out failures; the retry policy changed in 3.7.",
    "cryostat.md": "# Cryostat\n\nThe cryostat pressure alarm threshold is 1.2 bar; call the cryo expert.",
    "calorimeter.md": "# Calorimeter\n\nHigh-voltage trips of the hadronic calorimeter are reset from the slow control.",
    "network.md": "# Network\n\nTransfers to Tier-2 sites go through FTS; check the channel saturation first.",
    "shift.md": "# Shift\n\nThe shift leader signs the logbook at 07:00, 15:00 and 23:00 every day.",
    "magnet.md": "# Magnet\n\nThe solenoid is ramped down before any access to the experimental hall.",
}
QUESTIONS = [
    ("what is the cryostat alarm threshold", "cryostat.md"),
    ("how do I reset a calorimeter HV trip", "calorimeter.md"),
    ("when does the shift leader sign the logbook", "shift.md"),
    ("stage-out failures pilot restart", "pilot.md"),
    ("FTS channel saturation for Tier-2 transfers", "network.md"),
    ("solenoid ramp before hall access", "magnet.md"),
]


@pytest.fixture
def corpus(database):
    with make_client("authsvc") as client:
        caller = Caller(client, user("eval", ("app:access", "knowledge:read", "knowledge:write")))
        kb = caller.post("/kb/api/v1/knowledge-bases", json={"name": "Eval"}).json()["data"]["id"]
        names = {}
        for name, text in CORPUS.items():
            item = caller.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/manual",
                               json={"title": name, "content": text}).json()["data"]
            names[item["id"]] = name
        drain(client)
        yield caller, kb, names


def test_expected_document_in_top5(corpus):
    caller, kb, names = corpus
    misses = []
    for question, expected in QUESTIONS:
        results = caller.post("/kb/api/v1/knowledge-search",
                              json={"query": question, "knowledge_base_ids": [kb], "match_count": 5}).json()["data"]
        top = [names[r["knowledge_id"]] for r in results[:5]]
        if expected not in top:
            misses.append((question, expected, top))
    assert not misses, misses
