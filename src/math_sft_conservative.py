"""One matched lower-rate worked SFT experiment; never select on confirmation."""

import argparse
import shutil

from .common import ROOT, load_config, sha256_file, write_json
from .math_sft import OUT, PILOT, evaluate, fixed_split, paired, reviewed_data, train

CONFIG = ROOT / "configs/math_sft_conservative.yaml"
PUBLISHED = ROOT / "results/sft_conservative"
ORIGINAL = ROOT / "results/sft_followup"
LOCAL_ORIGINAL = PILOT / "work/math_sft"
FIELDS = ("correct", "previous", "strict", "source", "truncated", "n_tokens")


def prepare():
    if OUT == LOCAL_ORIGINAL:
        raise ValueError("Use an isolated LLM_MATH_SFT_OUT for the conservative run")
    old = load_config(ORIGINAL / "training_manifest.json")
    if not (LOCAL_ORIGINAL / "training.json").is_file():
        from .math_sft import prepare as prepare_data
        prepare_data()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name in ("training.json", "data_manifest.json"):
            source, target = LOCAL_ORIGINAL / name, OUT / name
            if not target.exists():
                shutil.copyfile(source, target)
    ids = [x["id"] for x in reviewed_data()]
    if sha256_file(OUT / "training.json") != old["dataset_sha256"] or ids != old["training_ids"]:
        raise ValueError("Training examples or order differ from the original run")
    if (LOCAL_ORIGINAL / "dev_base.json").is_file() and not (OUT / "dev_base.json").exists():
        path = LOCAL_ORIGINAL / "dev_base.json"
        if sha256_file(path) != old["local_raw_sha256"]["dev"]["base"]:
            raise ValueError("Previous base evaluation checksum differs")
        shutil.copyfile(path, OUT / path.name)
    write_json(OUT / "plan.json", {"policy": load_config(CONFIG), "training_ids": ids,
               "dataset_sha256": old["dataset_sha256"], "review_sha256": old["review_sha256"]})


def summarize(split, rows, cfg):
    n = len(rows)
    comparisons = {m: paired([r["base"][m] for r in rows], [r["conservative"][m] for r in rows])
                   for m in ("correct", "previous", "strict")}
    counts = {
        arm: {"unresolved": sum(r[arm]["source"] == "unresolved" for r in rows),
              "truncated": sum(r[arm]["truncated"] for r in rows),
              "mean_tokens": sum(r[arm]["n_tokens"] for r in rows) / n}
        for arm in ("base", "conservative")
    }
    lower, _ = comparisons["correct"]["ci_pp"]
    coverage = (max(x["unresolved"] for x in counts.values()) / n <= cfg["max_unresolved_fraction"]
                and (counts["conservative"]["unresolved"] - counts["base"]["unresolved"]) * 100 / n
                <= cfg["max_extra_unresolved_pp"])
    preserve = coverage and lower > -cfg["preservation_margin_pp"]
    improvement = (coverage and comparisons["correct"]["delta_pp"] >= cfg["min_gain_pp"]
                   and lower > 0 and comparisons["previous"]["delta_pp"] > 0
                   and comparisons["strict"]["delta_pp"] > 0)
    common = [r for r in rows if all(not r[a]["truncated"] and r[a]["source"] != "unresolved"
                                     for a in ("base", "conservative"))]
    return {"split": split, "n": n, "comparisons": comparisons, "counts": counts,
            "common_complete_resolved_descriptive": {
                "n": len(common), "base": sum(r["base"]["correct"] for r in common),
                "conservative": sum(r["conservative"]["correct"] for r in common)},
            "preservation_supported": preserve, "improvement_gate": improvement,
            "decision": ("confirm" if improvement else "stop") if split == "dev"
            else ("supported" if improvement else "inconclusive")}


def compare(split):
    if split == "confirmation" and load_config(OUT / "dev_comparison.json")["decision"] != "confirm":
        raise RuntimeError("Confirmation requires a positive dev gate")
    base = load_config(OUT / f"{split}_base.json")["rows"]
    worked = load_config(OUT / f"{split}_worked.json")["rows"]
    rows = []
    for q, a, b in zip(fixed_split(split), base, worked, strict=True):
        if any(x["id"] != q["id"] or x["question_sha256"] != q["question_sha256"] for x in (a, b)):
            raise ValueError("Evaluation identities differ")
        rows.append(dict(id=q["id"], question_sha256=q["question_sha256"],
                         base={k: a[k] for k in FIELDS}, conservative={k: b[k] for k in FIELDS}))
    cfg = load_config(OUT / "plan.json")["policy"]
    result = summarize(split, rows, cfg)
    write_json(OUT / f"{split}_paired.json", {"rows": rows})
    write_json(OUT / f"{split}_comparison.json", result)
    print(result, flush=True)


def verify(path):
    config = load_config(CONFIG)
    report = load_config(path / "run.json")
    old = load_config(ORIGINAL / "training_manifest.json")
    if (report["dataset_sha256"] != old["dataset_sha256"] or report["review_sha256"] != old["review_sha256"]
            or report["training_ids"] != old["training_ids"] or report["policy"] != config
            or report["training"]["final_global_step"] != old["arms"]["worked"]["final_global_step"]
            or report["learning_rate_comparison"] != {
                "original": load_config("configs/math_sft.yaml")["learning_rate"],
                "conservative": config["learning_rate"]}):
        raise ValueError("Conservative training provenance differs")
    split_index = load_config(PILOT / "math_split.json")["splits"]
    dev = load_config(path / "dev_comparison.json")
    for split in ("dev", "confirmation") if dev["decision"] == "confirm" else ("dev",):
        rows = load_config(path / f"{split}_paired.json")["rows"]
        if [x["id"] for x in rows] != split_index[split]["ids"]:
            raise ValueError("Held-out IDs differ")
        if summarize(split, rows, config) != load_config(path / f"{split}_comparison.json"):
            raise ValueError("Recomputed comparison differs")
        if split == "dev":
            prior = load_config(ORIGINAL / "dev_paired.json")["rows"]
            if [(x["id"], x["question_sha256"], x["base"]) for x in rows] != [
                    (x["id"], x["question_sha256"], x["base"]) for x in prior]:
                raise ValueError("Baseline comparison is not paired with the earlier run")
    if dev["decision"] == "stop" and (path / "confirmation_paired.json").exists():
        raise ValueError("Confirmation was opened despite a stopped dev gate")
    return dev


def record():
    status = load_config(OUT / "status.json")
    if status["status"] != "complete":
        raise ValueError("Run is incomplete")
    plan = load_config(OUT / "plan.json")
    recipe = load_config(OUT / "worked_manifest.json")
    old = load_config(ORIGINAL / "training_manifest.json")
    for key in ("dataset_sha256", "review_sha256", "model", "revision", "versions"):
        if recipe[key] != old[key]:
            raise ValueError(f"Original training provenance differs: {key}")
    if recipe["trainer"]["learning_rate"] != plan["policy"]["learning_rate"] or recipe["ids"] != plan["training_ids"]:
        raise ValueError("Executed training differs from the plan")
    original_rate = load_config("configs/math_sft.yaml")["learning_rate"]
    previous_recipe = LOCAL_ORIGINAL / "worked_manifest.json"
    if previous_recipe.is_file():
        previous_values = load_config(previous_recipe)["trainer"]
        new_values = recipe["trainer"]
        differences = {k for k in previous_values.keys() | new_values.keys()
                       if previous_values.get(k) != new_values.get(k)}
        if differences != {"output_dir", "learning_rate"} or previous_values["learning_rate"] != original_rate:
            raise ValueError(f"Unexpected trainer differences: {sorted(differences)}")
    PUBLISHED.mkdir(parents=True, exist_ok=True)
    splits = ("dev", "confirmation") if "compare_confirmation" in status["completed"] else ("dev",)
    for split in splits:
        for suffix in ("paired", "comparison"):
            name = f"{split}_{suffix}.json"
            shutil.copyfile(OUT / name, PUBLISHED / name)
    write_json(PUBLISHED / "run.json", {
        **plan, "model": recipe["model"], "revision": recipe["revision"], "versions": recipe["versions"],
        "learning_rate_comparison": {"original": original_rate, "conservative": plan["policy"]["learning_rate"]},
        "training": load_config(OUT / "worked_train.json"),
        "training_memory": load_config(OUT / "train_worked_memory.json"),
        "raw_sha256": {s: {a: sha256_file(OUT / f"{s}_{a}.json") for a in ("base", "worked")} for s in splits},
    })
    print(verify(PUBLISHED), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "eval", "compare", "record", "verify"))
    parser.add_argument("--resume")
    parser.add_argument("--split", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--arm", choices=("base", "worked"), default="worked")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        train("worked", args.resume, learning_rate=load_config(OUT / "plan.json")["policy"]["learning_rate"])
    elif args.action == "eval":
        evaluate(args.split, args.arm)
    elif args.action == "compare":
        compare(args.split)
    elif args.action == "record":
        record()
    else:
        print(verify(PUBLISHED))
