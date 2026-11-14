"""Record artifact and runtime provenance after the independent comparison."""

from importlib.metadata import version

from src.common import load_config, sha256_file, write_json
from src.math_independent import LABELS, REPORT, WORK, ready, verify


def record():
    plan, items = ready()
    state = load_config(WORK / "status.json")
    if state["status"] != "complete":
        raise RuntimeError("Independent run is incomplete")
    protocol_digest = sha256_file(REPORT / "protocol.json")
    raw = {}
    for label in LABELS:
        files = sorted(WORK.glob(f"raw_{label}_*.json"))
        rows = []
        raw[label] = []
        for path in files:
            shard = load_config(path)
            if shard["protocol"] != protocol_digest:
                raise ValueError("Raw shard belongs to a different protocol")
            raw[label].append({"file": path.name, "sha256": sha256_file(path),
                               "rows": len(shard["rows"]), "generation_seconds": shard["elapsed"]})
            rows.extend(shard["rows"])
        if [(x["id"], x["question_sha256"]) for x in rows] != [
                (x["id"], x["question_sha256"]) for x in items]:
            raise ValueError("Raw rows differ from the frozen test")
    verify()
    write_json(REPORT / "run.json", {
        "protocol_sha256": protocol_digest, "comparison_sha256": sha256_file(REPORT / "comparison.json"),
        "model_revision": plan["revision"], "versions": {p: version(p) for p in
            ("torch", "vllm", "transformers", "math-verify", "numpy", "scipy")},
        "local_raw_shards": raw, "memory": {a: load_config(WORK / f"generate_{a}_memory.json") for a in LABELS},
        "completion_utc": state["updated_utc"], "no_training": True,
    })


if __name__ == "__main__":
    record()
