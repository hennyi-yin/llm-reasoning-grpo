"""Regrade paired saved GSM8K responses without changing the original ledger."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.common import ROOT, write_json
from src.extract import numeric_equal
from src.followup_answers import final_candidate
from src.stats import mcnemar_exact


def run(output: Path, overrides_path: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    reviews = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path else {}
    rows = {"base": {}, "ablation": {}}
    used, queue = set(), {}
    with (ROOT / "results/per_problem.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["task"] != "gsm8k" or row["model"] not in rows:
                continue
            raw = (ROOT / row["raw_output_path"]).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != row["raw_output_sha256"]:
                raise ValueError(f"Saved answer hash mismatch: {row['id']}")
            item = json.loads(raw)
            if item["problem"]["id"] != row["id"] or item["identity"]["model"] != row["model"]:
                raise ValueError("Saved answer identity mismatch")
            text = item["generation"]["text"]
            candidate, source = final_candidate(text, numeric=True)
            # Review keys hash text only, so the reviewer sees neither label nor reference.
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in reviews:
                candidate = reviews[key]
                if candidate is not None and not isinstance(candidate, str):
                    raise ValueError(f"Review answer must be a string or null: {key}")
                used.add(key)
                source = "manual_final" if candidate is not None else "manual_no_final"
            elif source == "review":
                queue[key] = text
            rows[row["model"]][row["id"]] = {
                "raw_sha256": digest, "source": source, "candidate": candidate,
                "strict": row["correct"] == "True",
                "regraded": numeric_equal(candidate, item["problem"]["gold"]),
            }
    if used != set(reviews) or set(rows["base"]) != set(rows["ablation"]) or len(rows["base"]) != 1319:
        raise ValueError("Incomplete paired rows or stale review keys")

    def comparison(field):
        import numpy as np
        a = [x[field] for x in rows["base"].values()]
        b = [rows["ablation"][i][field] for i in rows["base"]]
        left, right, n = sum(a), sum(b), len(a)
        gains = sum(not x and y for x, y in zip(a, b))
        losses = sum(x and not y for x, y in zip(a, b))
        draws = np.random.default_rng(0).multinomial(n, [gains/n, losses/n, (n-gains-losses)/n], 10_000)
        ci = np.percentile(100 * (draws[:, 0] - draws[:, 1]) / n, [2.5, 97.5]).tolist()
        return {"base_correct": left, "grpo_correct": right, "base_pct": 100 * left / n,
                "grpo_pct": 100 * right / n, "delta_pp": 100 * (right - left) / n,
                "gains": gains, "losses": losses, "paired_bootstrap_95_ci_pp": ci,
                "mcnemar_p_descriptive": mcnemar_exact(a, b)}
    summary = {"n_pairs": len(rows["base"]), "pending_unique_reviews": len(queue),
               "manual_reviews": len(used), "strict": comparison("strict"),
               "regraded": comparison("regraded")}
    write_json(output / "summary.json", summary)
    write_json(output / "paired_rows.json", rows)
    write_json(output / "review_queue.json", dict(sorted(queue.items())))
    return summary


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=ROOT / "results/followup/gsm8k")
    cli.add_argument("--overrides", type=Path)
    args = cli.parse_args()
    print(json.dumps(run(args.out, args.overrides), indent=2))
