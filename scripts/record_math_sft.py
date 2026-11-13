"""Archive audit metadata and paired SFT judgments from a completed local run."""

import argparse
from collections import defaultdict
from pathlib import Path
import shutil

from scripts.verify_math_sft import verify
from src.common import ROOT, load_config, sha256_file, write_json
from src.math_sft import ARMS, CONFIG, OUT, PILOT, fixed_split, paired


def diagnostics(run):
    items = fixed_split("dev")
    outputs = {label: load_config(run / f"dev_{label}.json")["rows"] for label in ARMS}
    ledger = load_config(run / "dev_paired.json")["rows"]
    by_level, by_subject = defaultdict(list), defaultdict(list)
    for item, row in zip(items, ledger, strict=True):
        if item["id"] != row["id"] or item["question_sha256"] != row["question_sha256"]:
            raise ValueError("Diagnostic identities do not match")
        by_level[item["level"]].append(row)
        by_subject[item["subject"]].append(row)

    def strata(groups):
        return {key: {"n": len(rows), **{arm: sum(x[arm]["correct"] for x in rows) for arm in ARMS}}
                for key, rows in sorted(groups.items())}

    # This excludes any problem where at least one model was clipped or the
    # primary evaluator could not resolve the answer. Treat it as descriptive:
    # this subset depends on model outputs.
    common = [x for x in ledger if all(not x[a]["truncated"] and
                                     x[a]["source"] != "unresolved" for a in ARMS)]
    n = len(common)
    if not n:
        raise ValueError("No jointly resolved complete responses")
    return {
        "by_level_descriptive": strata(by_level), "by_subject_descriptive": strata(by_subject),
        "common_resolved_unclipped": {"n": n, **{a: sum(x[a]["correct"] for x in common) for a in ARMS}},
        "answer_only_vs_base": paired([x["base"]["correct"] for x in ledger],
                                     [x["short"]["correct"] for x in ledger]),
        "recovered_by_new_judge": {a: sum(r["correct"] and not r["previous"] for r in rows)
                                   for a, rows in outputs.items()},
        "worked_vs_base_changed_ids": {"gained": [x["id"] for x in ledger if x["worked"]["correct"]
                                                  and not x["base"]["correct"]],
                                      "lost": [x["id"] for x in ledger if x["base"]["correct"]
                                               and not x["worked"]["correct"]]},
    }


def record(run, out):
    status = load_config(run / "status.json")
    if status["status"] != "complete":
        raise ValueError("The SFT experiment has not finished")
    source = load_config(run / "data_manifest.json")
    arms = {label: load_config(run / f"{label}_train.json") for label in ARMS[1:]}
    recipes = {label: load_config(run / f"{label}_manifest.json") for label in arms}
    short, worked = recipes["short"], recipes["worked"]
    common = ("ids", "dataset_sha256", "review_sha256", "model", "revision", "versions")
    if any(short[k] != worked[k] for k in common):
        raise ValueError("SFT arm provenance differs")
    expected_review = ROOT / "results/sft_followup/reference_review.json"
    if short["review_sha256"] != sha256_file(expected_review):
        raise ValueError("Reference review changed after training")
    out.mkdir(parents=True, exist_ok=True)
    completed_splits = ["dev"]
    if "compare_confirmation" in status["completed"]:
        completed_splits.append("confirmation")
    for split in completed_splits:
        for file in (f"{split}_paired.json", f"{split}_comparison.json"):
            shutil.copyfile(run / file, out / file)
    probe = run / "dev_scale_probe.json"
    if probe.is_file():
        shutil.copyfile(probe, out / probe.name)
    write_json(out / "training_manifest.json", {
        "model": short["model"], "revision": short["revision"],
        "versions": short["versions"], "source_revision": source["source_revision"],
        "source_file_sha256": source["source_file_sha256"],
        "training_ids": short["ids"], "dataset_sha256": short["dataset_sha256"],
        "review_sha256": short["review_sha256"], "config_sha256": sha256_file(CONFIG),
        "eligible_sample_counts": source["subject_counts"], "filter_rejections": source["rejected"],
        "arms": {label: {
            **arms[label], "target_tokens_including_eos": recipes[label]["target_tokens"],
            **load_config(run / f"train_{label}_memory.json"),
        } for label in arms},
        "local_raw_sha256": {split: {label: sha256_file(run / f"{split}_{label}.json") for label in ARMS}
                             for split in completed_splits},
        "status": status["decision"],
    })
    write_json(out / "dev_diagnostics.json", diagnostics(run))
    return verify(out, PILOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=ROOT / "results/sft_followup")
    args = parser.parse_args()
    for split, metrics in record(args.run, args.out).items():
        print(split, metrics["decision"], metrics["counts"])
