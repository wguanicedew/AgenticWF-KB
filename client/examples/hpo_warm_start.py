"""HPO warm start with the KB's Python client (docs/hpo-warm-start.md).

    python hpo_warm_start.py upload      1) store some past HPO jobs as `hpo_job` records
    python hpo_warm_start.py suggest     2) ask for warm-start points for a new, similar job
    python hpo_warm_start.py optuna      both with Optuna: warm-start a study, run it, upload it
                                            (needs the hpo extra: pip install './client[hpo]')

The KB and its token come from the environment, as for `agenticwf kb` (KBClient.from_env, which
needs agenticwf-auth-client):

    # behind Caddy: `agenticwf login` (or AGENTICWF_TOKEN=pat_…) and AGENTICWF_API_URL, …
    # a standalone KB, with its own token:
    export AGENTICWF_KB_URL=http://127.0.0.1:8766/kb
    export AGENTICWF_KB_TOKEN=$(docker exec agenticwf-kb agenticwf-kb token)

Without agenticwf-auth-client, build the client yourself:
    KBClient("http://127.0.0.1:8766/kb", api_key=token)  or  KBClient("https://host/app/kb", bearer=token)

The KB needs the hpo plugin: `[records] plugins = hpo` in kb.cfg.
"""

from __future__ import annotations

import json
import math
import random
import sys

from agenticwf.kb.client import KBClient

KB_NAME = "HPO history"   # a records KB of yours; created on first use


def records_kb(kb: KBClient, name: str = KB_NAME) -> str:
    """The id of your records KB called `name`, created if missing."""
    for item in kb.knowledge_bases():
        if item["name"] == name and item["type"] == "records":
            return item["id"]
    return kb.request("POST", "/knowledge-bases", json={"name": name, "type": "records"})["id"]


# -- 1) upload past jobs --------------------------------------------------------------------------
def train(lr: float, weight_decay: float, batch_size: int, warmup_steps: int, dataset_shift: float = 0.0) -> float:
    """A stand-in for a training run: the validation loss, best near lr=3e-3, wd=0.02, bs=64."""
    return (0.6 + 0.08 * (math.log10(lr) - math.log10(3e-3) + dataset_shift) ** 2 + 4 * (weight_decay - 0.02) ** 2
            + 0.03 * abs(math.log2(batch_size / 64)) + 0.00002 * abs(warmup_steps - 500))


def past_job(job_id: str, description: str, dataset: str, n_trials: int, seed: int, shift: float) -> dict:
    """An hpo_job record: what the job searched, and every trial with its result."""
    space = {
        "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
        "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        "warmup_steps": {"type": "int", "low": 0, "high": 2000, "step": 100},
    }
    rng = random.Random(seed)
    trials = []
    for number in range(n_trials):
        params = {"lr": 10 ** rng.uniform(-5, -1), "weight_decay": rng.uniform(0, 0.1),
                  "batch_size": rng.choice([32, 64, 128]), "warmup_steps": 100 * rng.randint(0, 20)}
        trials.append({"number": number, "params": params, "value": round(train(**params, dataset_shift=shift), 4),
                       "status": "complete"})
    return {"job_id": job_id, "record_name": job_id, "description": description,
            "objective": {"name": "val_loss", "direction": "minimize"},
            "search_space": space, "trials": trials,
            "metadata": {"dataset": dataset, "model": "resnet50", "framework": "custom"}}


def upload(kb: KBClient) -> str:
    kb_id = records_kb(kb)
    jobs = [
        past_job("resnet50-cifar100-adamw", "ResNet-50 fine-tuning on CIFAR-100, AdamW, cosine schedule",
                 "cifar100", n_trials=40, seed=1, shift=0.0),
        past_job("resnet50-cifar10-adamw", "ResNet-50 fine-tuning on CIFAR-10, AdamW, cosine schedule",
                 "cifar10", n_trials=30, seed=2, shift=0.2),
    ]
    # Upsert by job_id: running this again updates the two records instead of adding new ones.
    report = kb.upsert_records(kb_id, jobs, record_type="hpo_job")
    print(f"KB {kb_id}: {json.dumps(report)}")
    return kb_id


# -- 2) suggest warm-start points for a new job ---------------------------------------------------
NEW_SPACE = {   # the new job: a narrower lr range, no warmup_steps, a new parameter (dropout)
    "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "weight_decay": {"type": "float", "low": 0.0, "high": 0.05},
    "batch_size": {"type": "categorical", "choices": [64, 128, 256]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.1},
}


def suggest(kb: KBClient) -> dict:
    kb_id = records_kb(kb)
    data = kb.suggest(
        knowledge_base_ids=[kb_id],                          # omit: every records KB you can read
        objective={"name": "val_loss", "direction": "minimize"},
        description="ResNet-50 fine-tuning on Tiny ImageNet, AdamW",
        search_space=NEW_SPACE,
        n=5,                                                 # points from similar past jobs
        n_total=8,                                           # padded with space-filling (Halton) points
        filter={"field": "metadata.model", "op": "eq", "value": "resnet50"},
        options={"min_similarity": 0.3, "seed": 0},
    )
    for job in data["source_jobs"]:
        print(f"source job {job['job_id']}: similarity {job['similarity']:.2f} {job['breakdown']}")
    print(f"parameters no past job has: {data['unmatched_params']}")
    for point in data["points"]:
        source = point["source"]
        origin = (f"from {source['job_id']} trial {source['trial_index']} (val_loss {source['value']}, "
                  f"score {point['score']:.2f}, filled {point.get('filled_params')})"
                  if "job_id" in source else source.get("type", ""))
        print(f"  {json.dumps(point['params'])}  {origin}")
    return data


# -- both with Optuna ---------------------------------------------------------------------------
def with_optuna(kb: KBClient) -> None:
    import optuna

    from agenticwf.kb.client.hpo import upload_study, warm_start

    kb_id = records_kb(kb)

    def objective(trial) -> float:
        params = {"lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                  "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.05),
                  "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256])}
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        return train(**params, warmup_steps=500, dataset_shift=0.1) + 0.1 * (dropout - 0.2) ** 2

    study = optuna.create_study(direction="minimize")
    data = warm_start(kb, study, NEW_SPACE, [kb_id], objective_name="val_loss",
                      description="ResNet-50 fine-tuning on Tiny ImageNet, AdamW", n=5, n_total=8)
    print(f"enqueued {len(data['points'])} warm-start trials")   # Optuna runs them first
    study.optimize(objective, n_trials=30)
    print(f"best val_loss {study.best_value:.4f} with {study.best_params}")
    # The finished job becomes history for the next one.
    report = upload_study(kb, study, kb_id, job_id="resnet50-tinyimagenet-adamw", search_space=NEW_SPACE,
                          objective_name="val_loss", description="ResNet-50 fine-tuning on Tiny ImageNet, AdamW",
                          metadata={"dataset": "tinyimagenet", "model": "resnet50", "framework": "optuna"})
    print(f"uploaded: {json.dumps(report)}")


def main() -> None:
    steps = {"upload": upload, "suggest": suggest, "optuna": with_optuna}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        sys.exit(__doc__)
    with KBClient.from_env() as kb:
        steps[sys.argv[1]](kb)


if __name__ == "__main__":
    main()
