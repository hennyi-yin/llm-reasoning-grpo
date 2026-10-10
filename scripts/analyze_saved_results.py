"""Describe paired boxed-answer transitions using saved final records only."""
from collections import Counter, defaultdict
import csv
import json

from src.common import ROOT, sha256_file, write_json
from src.stats import COMPARISONS, EXPECTED_COUNTS, MODEL_LABELS, paired_comparison, parse_bool


def main():
    path = ROOT / "results/per_problem.csv"
    summary_path = ROOT / "results/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "measured":
        raise RuntimeError("Complete the final evaluation before interpreting it")
    groups = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = groups[row["model"], row["task"]]
            if row["id"] in group or row["split"] != "test":
                raise ValueError("Duplicate identity or non-test result")
            group[row["id"]] = row
    expected = {(m, t): n for m in MODEL_LABELS for t, n in EXPECTED_COUNTS.items()}
    if {key: len(value) for key, value in groups.items()} != expected:
        raise RuntimeError("Incomplete model/task groups")
    observations = []
    for task in EXPECTED_COUNTS:
        for a, b in COMPARISONS:
            left, right = groups[a, task], groups[b, task]
            comparison = paired_comparison(list(left.values()), list(right.values()))
            frozen = next(r for r in summary["comparisons"]
                          if (r["task"], r["a"], r["b"]) == (task, a, b))
            if any(comparison[k] != frozen[k] for k in comparison):
                raise RuntimeError("Paired counts differ from the frozen final report")
            transitions = Counter()
            for identity in left:
                x, y = left[identity], right[identity]
                correct_a, correct_b = parse_bool(x["correct"]), parse_bool(y["correct"])
                outcome = "gain" if not correct_a and correct_b else (
                    "loss" if correct_a and not correct_b else "unchanged")
                transitions[outcome, parse_bool(x["has_boxed"]), parse_bool(y["has_boxed"])] += 1
            observations.append({"task": task, "a": a, "b": b, **comparison,
                                 "boxed_transitions": [
                                     {"outcome": outcome, "a_has_boxed": ba,
                                      "b_has_boxed": bb, "n": count}
                                     for (outcome, ba, bb), count in sorted(transitions.items())]})
    evidence = {"status": "derived_from_saved_results", "new_generation": False,
                "interpretation": "Descriptive post-hoc format analysis; not a causal decomposition or new selection rule",
                "source_sha256": {"results/per_problem.csv": sha256_file(path),
                                  "results/summary.json": sha256_file(summary_path)},
                "comparisons": observations}
    write_json("results/format_analysis.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
