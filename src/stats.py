"""Reproducible accuracy intervals and identity-checked paired comparisons."""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

MODEL_LABELS = ("base", "sft", "grpo", "ablation", "quantized")
COMPARISONS = (("base", "sft"), ("sft", "grpo"), ("base", "grpo"),
               ("ablation", "grpo"), ("quantized", "grpo"))
EXPECTED_COUNTS = {"gsm8k": 1319, "math500": 500}


def _binary(values):
    import numpy as np
    x = np.asarray(values)
    if x.ndim != 1 or not len(x) or not np.isin(x, [0, 1]).all():
        raise ValueError("correctness must be a nonempty one-dimensional binary array")
    return x.astype(bool)


def bootstrap_ci(correct, n_boot: int = 10_000, seed: int = 0) -> list[float]:
    import numpy as np
    x = _binary(correct).astype(float)
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    # Bound memory while preserving the same draws as one 10,000 x n array.
    for start in range(0, n_boot, 1000):
        count = min(1000, n_boot - start)
        means[start:start + count] = rng.choice(x, (count, len(x))).mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def mcnemar_exact(a, b) -> float:
    from scipy.stats import binomtest
    aa, bb = _binary(a), _binary(b)
    if len(aa) != len(bb):
        raise ValueError("paired arrays must have identical lengths")
    n01, n10 = int((~aa & bb).sum()), int((aa & ~bb).sum())
    return float(binomtest(n10, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0


def parse_bool(value) -> bool:
    if value in (True, 1, "1", "True", "true"):
        return True
    if value in (False, 0, "0", "False", "false"):
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def paired_comparison(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Align by stable problem ID, then require identical question and protocol."""
    def index(rows):
        output = {}
        for row in rows:
            key = (row["task"], str(row["id"]))
            if key in output:
                raise ValueError(f"duplicate paired problem identity: {key}")
            output[key] = row
        if not output:
            raise ValueError("paired comparison requires nonempty rows")
        return output
    a, b = index(rows_a), index(rows_b)
    if a.keys() != b.keys():
        raise ValueError("paired models must contain exactly the same task/problem IDs")
    keys = sorted(a)
    for key in keys:
        for field in ("question_sha256", "protocol_sha256", "split"):
            if not a[key].get(field) or a[key][field] != b[key].get(field):
                raise ValueError(f"paired identity mismatch for {key}: {field}")
    aa = [parse_bool(a[key]["correct"]) for key in keys]
    bb = [parse_bool(b[key]["correct"]) for key in keys]
    n01 = sum(not x and y for x, y in zip(aa, bb))
    n10 = sum(x and not y for x, y in zip(aa, bb))
    return {"n": len(keys), "n01": n01, "n10": n10,
            "difference_pp": 100 * (sum(bb) - sum(aa)) / len(keys),
            "p_value": mcnemar_exact(aa, bb)}


def summarize_rows(rows: list[dict], *, n_boot: int = 10_000, seed: int = 0) -> dict:
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["model"], row["task"], str(row["id"]))
        if key in seen:
            raise ValueError(f"duplicate result identity: {key}")
        seen.add(key)
        if row.get("split") != "test":
            raise ValueError("final summary accepts test rows only")
        groups[key[:2]].append(row)
    accuracy, comparisons = [], []
    for (model, task), group in sorted(groups.items()):
        protocols = {r.get("protocol_sha256") for r in group}
        identities = {r.get("model_sha256") for r in group}
        if len(protocols) != 1 or None in protocols or len(identities) != 1 or None in identities:
            raise ValueError(f"mixed or missing provenance in {model}/{task}")
        correct = [parse_bool(r["correct"]) for r in group]
        n = len(group)
        accuracy.append({"model": model, "task": task, "n": n,
                         "accuracy": sum(correct) / n,
                         "ci95": bootstrap_ci(correct, n_boot, seed),
                         "boxed_rate": sum(parse_bool(r["has_boxed"]) for r in group) / n,
                         "mean_tokens": sum(int(r["n_tokens"]) for r in group) / n,
                         "truncated_rate": sum(parse_bool(r["truncated"]) for r in group) / n})
    for task in EXPECTED_COUNTS:
        for a, b in COMPARISONS:
            if (a, task) in groups and (b, task) in groups:
                comparisons.append({"task": task, "a": a, "b": b,
                                    **paired_comparison(groups[a, task], groups[b, task])})
    complete = all(len(groups[label, task]) == n for label in MODEL_LABELS
                   for task, n in EXPECTED_COUNTS.items())
    return {"status": "measured" if complete else "incomplete", "accuracy": accuracy,
            "comparisons": comparisons, "bootstrap_resamples": n_boot,
            "bootstrap_seed": seed, "ci_level": 0.95,
            "difference_direction": "b minus a in percentage points"}


def summarize_file(path, *, n_boot: int = 10_000, seed: int = 0) -> dict:
    path = Path(path)
    if not path.exists():
        return summarize_rows([], n_boot=n_boot, seed=seed)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return summarize_rows(rows, n_boot=n_boot, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/per_problem.csv")
    parser.add_argument("--output", default="results/accuracy_summary.json")
    args = parser.parse_args()
    summary = summarize_file(args.input)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
