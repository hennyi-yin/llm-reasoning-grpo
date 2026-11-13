"""Recompute the saved MATH dev result without model or dataset downloads."""

import argparse
from pathlib import Path

import numpy as np

from src.common import ROOT, load_config, sha256_file
from src.stats import mcnemar_exact


def verify(root: Path, *, local_raw: bool = False) -> dict:
    paired = load_config(root / "math_dev_paired.json")
    split = load_config(root / "math_split.json")
    screen = load_config(root / "math_screen.json")
    cfg = load_config("configs/math_pilot.yaml")
    dev = split["splits"]["dev"]
    rows = paired["rows"]
    ids = [row["id"] for row in rows]
    id_sets = {name: set(entry["ids"]) for name, entry in split["splits"].items()}
    n = len(rows)
    if (ids != dev["ids"] or paired["split_sha256"] != dev["sha256"]
            or len(set(ids)) != n or n != cfg["data"]["dev_count"]
            or len(set.union(*id_sets.values())) != sum(map(len, id_sets.values()))):
        raise ValueError("Dev ledger or split identity mismatch")
    selected = screen["selected_ids"]
    if (len(set(selected)) != len(selected) or not set(selected) <= id_sets["screen"]
            or sum(screen["groups"].values()) + len(screen["invalid_reference_ids"]) != screen["screened"]):
        raise ValueError("Screening evidence does not match the training split")
    train = load_config(root / "math_train_summary.json")
    if (train["final_global_step"] != cfg["training"]["max_steps"]
            or len(train["step_seconds"]) != train["steps_completed"]):
        raise ValueError("Training summary does not reach the planned step")
    labels = ("base", "pilot")
    metrics = {}
    for key in ("correct", "strict"):
        a, b = ([row[label][key] for row in rows] for label in labels)
        gains = sum(not x and y for x, y in zip(a, b))
        losses = sum(x and not y for x, y in zip(a, b))
        draws = np.random.default_rng(0).multinomial(n, [gains/n, losses/n, (n-gains-losses)/n], 10_000)
        metrics[key] = {
            "base": sum(a), "pilot": sum(b), "delta_pp": 100 * (gains - losses) / n,
            "gains": gains, "losses": losses,
            "paired_bootstrap_95_ci_pp": np.percentile(
                100 * (draws[:, 0] - draws[:, 1]) / n, [2.5, 97.5]).tolist(),
            "paired_p_descriptive": mcnemar_exact(a, b),
        }
    unknown = [sum(row[label]["source"] == "unresolved" for row in rows) for label in labels]
    truncated = [sum(row[label]["truncated"] for row in rows) for label in labels]
    policy = cfg["evaluation"]
    confirm = (metrics["correct"]["delta_pp"] >= policy["min_dev_gain_pp"]
               and max(unknown) / n <= policy["max_unresolved_fraction"]
               and 100 * abs(unknown[1] - unknown[0]) / n <= policy["max_unknown_delta_pp"])
    result = dict(n=n, split="dev", metrics=metrics, unresolved_base_pilot=unknown,
                  truncated_base_pilot=truncated, decision="confirm" if confirm else "stop")
    if result != load_config(root / "math_dev_comparison.json"):
        raise ValueError("Recomputed MATH dev result differs from the saved comparison")
    if local_raw:
        verify_raw(root, paired, screen)
    return result


def verify_raw(root, paired, screen):
    from src import followup_answers, math_pilot
    from src.evaluate import judge

    if (sha256_file(followup_answers.__file__) != screen["extractor_sha256"]
            or sha256_file(math_pilot.__file__) != screen["scorer_sha256"]):
        raise ValueError("Scorer differs from the one used for the saved run")
    work = root / "work"
    split_file = work / "math_dev.json"
    if sha256_file(split_file) != paired["split_sha256"]:
        raise ValueError("Local dev split checksum differs")
    evidence = load_config(root / "math_pilot_audit.json")
    items = load_config(split_file)["items"]
    for label in ("base", "pilot"):
        path = work / f"dev_{label}.json"
        if sha256_file(path) != evidence["local_raw_sha256"][label]:
            raise ValueError(f"Raw generation checksum differs: {label}")
        outputs = load_config(path)["rows"]
        if not len(items) == len(outputs) == len(paired["rows"]):
            raise ValueError(f"Raw generation count differs: {label}")
        for item, output, row in zip(items, outputs, paired["rows"]):
            if (item["id"] != output["id"] or item["id"] != row["id"]
                    or item["question_sha256"] != output["question_sha256"]
                    or item["question_sha256"] != row["question_sha256"]):
                raise ValueError("Raw generation identity differs")
            scored = row[label]
            _, source, correct = math_pilot.score(item["gold"], output["text"], truncated=output["truncated"])
            strict = judge("math500", item["gold"], output["text"])
            expected = dict(source=source, correct=correct, strict=strict,
                            n_tokens=output["n_tokens"], truncated=output["truncated"])
            if expected != scored:
                raise ValueError(f"Raw rejudge differs: {label}/{row['id']}")


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/followup")
    parser.add_argument("--local-raw", action="store_true", help="Also hash-check and rejudge local generations")
    args = parser.parse_args()
    print(json.dumps(verify(args.out, local_raw=args.local_raw), indent=2))
