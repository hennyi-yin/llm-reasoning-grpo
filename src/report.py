"""Generate result tables, resume A-G, and figures solely from measured artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .benchmark_common import digest, offline_speedup, rooted
from .bench_online import parse_weight_memory, validate_result
from .common import ROOT, sha256_file, write_json
from .stats import EXPECTED_COUNTS, MODEL_LABELS, parse_bool, summarize_file


def read_optional(path: str | Path) -> dict | None:
    file = rooted(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else None


def online_rows(precision: str) -> tuple[list[dict], float | None]:
    meta = read_optional(f"results/online_{precision}.json")
    if not meta or meta.get("status") != "measured":
        return [], None
    rows = []
    expected = [1, 8, 32] if precision == "bf16" else [1, 32]
    if [r["concurrency"] for r in meta["results"]] != expected:
        raise ValueError(f"Incomplete online {precision} concurrency scan")
    for record in meta["results"]:
        raw_path = rooted(record["raw_result"])
        if sha256_file(raw_path) != record["raw_sha256"]:
            raise ValueError(f"Raw benchmark result changed: {raw_path}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if raw.get("precision") != precision or raw.get("max_concurrency") != record["concurrency"]:
            raise ValueError("Raw serving benchmark precision/concurrency differs from metadata")
        for key in ("model_weights_sha256", "model_sha256", "protocol_sha256"):
            if not meta.get(key) or raw.get(key) != meta[key]:
                raise ValueError(f"Raw serving benchmark identity mismatch: {key}")
        rows.append({"engine": "vllm-online", "precision": precision,
                     "concurrency": record["concurrency"], "source": record["raw_result"],
                     **validate_result(raw)})
    log = rooted(meta["server_log"])
    if sha256_file(log) != meta["server_log_sha256"]:
        raise ValueError("vLLM startup log changed")
    return rows, parse_weight_memory(log.read_text(encoding="utf-8"))


def weight_ratio(metadata: dict | None) -> float | None:
    if not metadata or metadata.get("status") != "measured":
        return None
    ids = metadata.get("calibration_ids", [])
    if metadata.get("calibration_split") != "train" or len(ids) != 256 or len(set(ids)) != 256:
        raise ValueError("Invalid quantization calibration provenance")
    sizes = []
    for key in ("source", "compressed"):
        manifest = metadata[key]
        entries = manifest["files"]
        if not entries or digest(entries) != manifest["weights_sha256"]:
            raise ValueError("Invalid checkpoint weight manifest")
        total = sum(int(e["bytes"]) for e in entries)
        if total <= 0 or total != manifest["weight_bytes"]:
            raise ValueError("Invalid weight-only checkpoint byte count")
        sizes.append(total)
    return sizes[0] / sizes[1]


def complete_accuracy(summary: dict, model: str, task: str) -> float | None:
    for row in summary["accuracy"]:
        if row["model"] == model and row["task"] == task and row["n"] == EXPECTED_COUNTS[task]:
            return 100 * row["accuracy"]
    return None


def validate_provenance(hf: dict | None, vllm: dict | None, quant: dict | None) -> None:
    """Check actual raw-output hashes and tie every final artifact to one model."""
    identities: dict[str, set[str]] = {label: set() for label in MODEL_LABELS}
    protocols = set()
    csv_path = ROOT / "results/per_problem.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_path = rooted(row["raw_output_path"])
                if sha256_file(raw_path) != row["raw_output_sha256"]:
                    raise ValueError(f"Raw evaluation output changed: {raw_path}")
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                identity = raw["identity"]
                if row["model"] not in identities:
                    raise ValueError("Unknown final model label in the evaluation CSV")
                for key in ("model", "task", "split", "model_sha256", "protocol_sha256"):
                    if row[key] != identity.get(key):
                        raise ValueError(f"Evaluation CSV/raw identity mismatch: {key}")
                if str(raw["problem"]["id"]) != row["id"]:
                    raise ValueError("Evaluation CSV/raw problem ID mismatch")
                question_sha = hashlib.sha256(raw["problem"]["question"].encode("utf-8")).hexdigest()
                if question_sha != row["question_sha256"]:
                    raise ValueError("Evaluation CSV/raw question mismatch")
                from .evaluate import judge
                from .extract import last_boxed
                generation = raw["generation"]
                boxed = last_boxed(generation["text"])
                derived = {
                    "correct": judge(row["task"], raw["problem"]["gold"], generation["text"]),
                    "has_boxed": boxed is not None,
                    "truncated": generation.get("finish_reason") == "length"}
                if any(parse_bool(row[key]) != value for key, value in derived.items()):
                    raise ValueError("Evaluation CSV correctness/format/truncation differs from raw generation")
                if int(row["n_tokens"]) != len(generation["token_ids"]):
                    raise ValueError("Evaluation CSV token count differs from raw generation")
                if row.get("extracted_answer", "") != (boxed if boxed is not None else ""):
                    raise ValueError("Evaluation CSV extracted answer differs from raw generation")
                if row.get("finish_reason", "") != (generation.get("finish_reason") or ""):
                    raise ValueError("Evaluation CSV finish reason differs from raw generation")
                protocols.add(row["protocol_sha256"])
                if row["model"] in identities:
                    identities[row["model"]].add(row["model_sha256"])
    for record in (hf, vllm):
        if record and record.get("status") == "measured":
            if not record.get("model_sha256") or not record.get("protocol_sha256"):
                raise ValueError("Offline benchmark is missing evaluator provenance")
            identities["grpo"].add(record["model_sha256"])
            protocols.add(record["protocol_sha256"])
    for precision, label in (("bf16", "grpo"), ("w4a16", "quantized")):
        record = read_optional(f"results/online_{precision}.json")
        if record and record.get("status") == "measured":
            if not record.get("model_sha256") or not record.get("protocol_sha256"):
                raise ValueError("Online benchmark is missing evaluator provenance")
            identities[label].add(record["model_sha256"])
            protocols.add(record["protocol_sha256"])
    if quant and quant.get("status") == "measured":
        from .data import load_split
        train_ids = {f"train-{i}" for i in load_split()["train"]}
        calibration = quant.get("calibration_ids", [])
        if not set(calibration) <= train_ids:
            raise ValueError("Quantization calibration contains nontraining question IDs")
        if not quant.get("source_model_sha256") or not quant.get("compressed_model_sha256"):
            raise ValueError("Quantization lacks evaluator model identities")
        identities["grpo"].add(quant["source_model_sha256"])
        identities["quantized"].add(quant["compressed_model_sha256"])
        protocols.add(quant["protocol_sha256"])
    if len(protocols) > 1 or any(not value for value in protocols):
        raise ValueError("Accuracy, serving and quantization do not share the frozen protocol")
    if protocols:
        from .protocol import verify_frozen
        verify_frozen()
        if protocols != {sha256_file(ROOT / "docs/frozen_manifest.json")}:
            raise ValueError("Measured artifacts do not match the current frozen protocol")
    if any(len(values) > 1 for values in identities.values()):
        raise ValueError("Accuracy, serving and quantization reference different final model artifacts")


def assemble(root: Path = ROOT) -> dict:
    if root != ROOT:
        raise ValueError("Report root must be the current repository")
    accuracy = summarize_file(ROOT / "results/per_problem.csv")
    hf = read_optional("results/offline_hf.json")
    vllm = read_optional("results/offline_vllm.json")
    online_bf16, memory_bf16 = online_rows("bf16")
    online_w4, memory_w4 = online_rows("w4a16")
    quant = read_optional("results/quantization.json")
    validate_provenance(hf, vllm, quant)
    values = {"A": complete_accuracy(accuracy, "base", "gsm8k"),
              "B": complete_accuracy(accuracy, "grpo", "gsm8k"),
              "C": complete_accuracy(accuracy, "base", "math500"),
              "D": complete_accuracy(accuracy, "grpo", "math500"),
              "E": offline_speedup(hf, vllm), "F": None, "G": weight_ratio(quant)}
    q_accuracy = complete_accuracy(accuracy, "quantized", "gsm8k")
    f_test = next((r for r in accuracy["comparisons"] if r["task"] == "gsm8k"
                   and r["a"] == "quantized" and r["b"] == "grpo"
                   and r["n"] == EXPECTED_COUNTS["gsm8k"]), None)
    if values["B"] is not None and q_accuracy is not None and f_test:
        values["F"] = f_test["difference_pp"]
    ready = (accuracy["status"] == "measured" and all(v is not None for v in values.values())
             and len(online_bf16) == 3 and len(online_w4) == 2)
    return {"status": "measured" if ready else "pending_gpu_execution",
            "accuracy": accuracy["accuracy"], "comparisons": accuracy["comparisons"],
            "statistics": {k: v for k, v in accuracy.items() if k not in {"accuracy", "comparisons"}},
            "offline": {"hf": hf, "vllm": vllm}, "online": online_bf16 + online_w4,
            "quantization": quant,
            "vllm_weight_memory_gib": {"bf16": memory_bf16, "w4a16": memory_w4},
            "resume_values": values,
            "resume_value_units": {"A": "%", "B": "%", "C": "%", "D": "%", "E": "ratio",
                                   "F": "percentage points (GRPO bf16 minus W4A16)", "G": "weight-file size ratio"},
            "quantization_mcnemar_p": f_test["p_value"] if f_test else None,
            "sources": {"per_problem_csv": "results/per_problem.csv",
                        "offline_hf": "results/offline_hf.json", "offline_vllm": "results/offline_vllm.json",
                        "quantization": "results/quantization.json"}}


def fmt(value, precision=2) -> str:
    if value is None:
        return "Pending"
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Refusing to render a nonfinite result")
    return f"{value:.{precision}f}"


def accuracy_cell(row: dict | None) -> str:
    if not row:
        return "Pending"
    low, high = row["ci95"]
    suffix = f"; partial n={row['n']}" if row["n"] != EXPECTED_COUNTS[row["task"]] else ""
    return f"{100 * row['accuracy']:.2f}% [{100 * low:.2f}, {100 * high:.2f}]{suffix}"


def render_markdown(summary: dict) -> str:
    values = summary["resume_values"]
    if summary["status"] == "measured":
        opening = (f"GSM8K: base {fmt(values['A'])}% → SFT + GRPO {fmt(values['B'])}%; "
                   f"MATH-500: {fmt(values['C'])}% → {fmt(values['D'])}%. "
                   f"vLLM / strongest fitting HF throughput: {fmt(values['E'])}×.")
    else:
        opening = "GPU experiments are pending. No accuracy, speedup, or memory reduction is claimed."
    lines = [opening, "", "| Model | GSM8K accuracy (95% CI) | MATH-500 accuracy (95% CI) | Boxed GSM / MATH | Mean tokens GSM / MATH | Truncated GSM / MATH |",
             "|---|---|---|---|---|---|"]
    lookup = {(r["model"], r["task"]): r for r in summary["accuracy"]}
    for model in MODEL_LABELS:
        gsm, mathrow = lookup.get((model, "gsm8k")), lookup.get((model, "math500"))
        cells = []
        for key in ("boxed_rate", "mean_tokens", "truncated_rate"):
            scale = 1 if key == "mean_tokens" else 100
            suffix = "" if key == "mean_tokens" else "%"
            cells.append(" / ".join(fmt(r[key] * scale) + suffix if r else "Pending"
                                     for r in (gsm, mathrow)))
        lines.append(f"| {model} | {accuracy_cell(gsm)} | {accuracy_cell(mathrow)} | " + " | ".join(cells) + " |")
    lines += ["", "Bootstrap: 10,000 resamples, seed 0. Each complete task uses 1,319 GSM8K or 500 MATH-500 questions.", "",
              "| Comparison (b - a) | Task | n | Difference (pp) | Exact McNemar p | n01 / n10 |",
              "|---|---|---:|---:|---:|---:|"]
    for r in summary["comparisons"]:
        lines.append(f"| {r['b']} - {r['a']} | {r['task']} | {r['n']} | {fmt(r['difference_pp'])} | {r['p_value']:.6g} | {r['n01']} / {r['n10']} |")
    if not summary["comparisons"]:
        lines.append("| Pending | — | — | — | — | — |")
    lines += ["", "| Engine | Precision | Batch / concurrency | Output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |",
              "|---|---|---:|---:|---:|---:|"]
    hf, vllm = summary["offline"]["hf"], summary["offline"]["vllm"]
    if hf and hf.get("status") == "measured":
        for r in hf["batch_results"]:
            value = fmt(r.get("output_tokens_per_second")) if r["status"] == "measured" else "OOM"
            lines.append(f"| HF offline | bf16 | batch {r['batch_size']} | {value} | — | — |")
    else:
        lines.append("| HF offline (best of 1 / 8 / 16 / 32) | bf16 | Pending | Pending | — | — |")
    lines.append(f"| vLLM offline | bf16 | continuous batching | {fmt(vllm.get('output_tokens_per_second')) if vllm and vllm.get('status') == 'measured' else 'Pending'} | — | — |")
    for r in summary["online"]:
        lines.append(f"| vLLM online | {r['precision']} | {r['concurrency']} | {fmt(r['output_throughput'])} | {fmt(r['p50_ttft_ms'])} / {fmt(r['p99_ttft_ms'])} | {fmt(r['p50_tpot_ms'])} / {fmt(r['p99_tpot_ms'])} |")
    if not summary["online"]:
        lines.append("| vLLM online | bf16 / W4A16 | Pending | Pending | Pending | Pending |")
    lines += ["", "Offline: identical 512 training prompts, exactly 256 output tokens, one warmup and median of three runs. "
              "Online: official vLLM CLI, 500 synthetic requests, exactly 256 input / output tokens.", "",
              f"E = {fmt(values['E'])}×; G = {fmt(values['G'])}× (checkpoint weight files only); "
              f"F = {fmt(values['F'])} pp; quantization McNemar p = {fmt(summary['quantization_mcnemar_p'], 6)}.", "",
              "G measures serialized weight-file size. vLLM weight memory below is read separately from startup logs.", "",
              "| Precision | Checkpoint weight bytes | vLLM loaded-weight GiB |",
              "|---|---:|---:|"]
    q = summary["quantization"]
    for precision, key in (("bf16", "source"), ("w4a16", "compressed")):
        size = str(q[key]["weight_bytes"]) if q and q.get("status") == "measured" else "Pending"
        lines.append(f"| {precision} | {size} | {fmt(summary['vllm_weight_memory_gib'][precision], 3)} |")
    lines += ["", "| Resume value | A (%) | B (%) | C (%) | D (%) | E (×) | F (pp) | G (×) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|",
              "| Measured | " + " | ".join(fmt(values[k]) for k in "ABCDEFG") + " |", ""]
    for name in summary.get("figures", []):
        lines += [f"![Measured experiment curve]({name})", ""]
    lines += ["Generated by `python -m src.report` from measured result files. Missing data stays `null` in `results/summary.json`.", ""]
    return "\n".join(lines)


def plot_measured(summary: dict) -> list[str]:
    reward_series, dev_series = [], []
    for stage in ("sft", "grpo", "grpo_no_sft"):
        log = ROOT / "logs" / stage / "metrics.jsonl"
        if log.is_file():
            records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            for key in ("rewards/correctness_reward/mean", "reward", "completions/clipped_ratio", "frac_reward_zero_std"):
                values = [(r["step"], r[key]) for r in records if key in r and isinstance(r[key], (int, float))]
                if values:
                    reward_series.append((f"{stage}: {key}", values))
        history = ROOT / "results/dev" / stage / "selection_history.json"
        if history.is_file():
            records = json.loads(history.read_text(encoding="utf-8"))
            values = [(r["step"], 100 * r["accuracy"]) for r in records if r.get("split") == "dev"]
            if values:
                dev_series.append((stage, values))
    throughput_series = []
    for precision in ("bf16", "w4a16"):
        values = [(r["concurrency"], r["output_throughput"]) for r in summary["online"] if r["precision"] == precision]
        if values:
            throughput_series.append((precision, values))
    if not any((reward_series, dev_series, throughput_series)):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    for name, series, x_label, y_label in (
        ("training_metrics", reward_series, "Optimizer step", "Logged metric"),
        ("dev_accuracy", dev_series, "Checkpoint step", "Dev accuracy (%)"),
        ("throughput_concurrency", throughput_series, "Concurrency", "Output tokens / second")):
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, points in series:
            points = sorted(points)
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
        ax.set(xlabel=x_label, ylabel=y_label)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        relative = f"figures/{name}.png"
        (ROOT / "figures").mkdir(exist_ok=True)
        fig.savefig(ROOT / relative, dpi=160)
        plt.close(fig)
        figures.append(relative)
    return figures


def update_readme(markdown: str) -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    before, rest = text.split(begin)
    _, after = rest.split(end)
    path.write_text(before + begin + "\n\n" + markdown + "\n" + end + after, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summary = assemble()
    summary["figures"] = [] if args.no_plots else plot_measured(summary)
    write_json("results/summary.json", summary)
    markdown = render_markdown(summary)
    (ROOT / "RESULTS.md").write_text(markdown, encoding="utf-8")
    if args.update_readme:
        update_readme(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
