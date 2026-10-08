"""CPU-safe benchmark bookkeeping; no timings are invented or extrapolated."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .common import ROOT, sha256_file


def rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def weight_files(path: str | Path) -> list[Path]:
    """Count exactly the checkpoint's referenced weights, excluding optimizer etc."""
    folder = rooted(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing merged checkpoint: {folder}")
    if (folder / "adapter_config.json").exists():
        raise ValueError("Serving requires a merged checkpoint, not a LoRA adapter")
    for basename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = folder / basename
        if index.exists():
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            paths = [folder / name for name in sorted(set(mapping.values()))]
            if any(not p.is_file() or p.resolve().parent != folder.resolve() for p in paths):
                raise ValueError("Weight index contains a missing or out-of-directory shard")
            return paths
    for basename in ("model.safetensors", "pytorch_model.bin"):
        candidate = folder / basename
        if candidate.is_file():
            return [candidate]
    raise ValueError(f"No standard HF weight files in {folder}")


def checkpoint_manifest(path: str | Path) -> dict:
    files = weight_files(path)
    entries = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
               for p in files]
    return {"path": str(rooted(path)), "files": entries,
            "weight_bytes": sum(p["bytes"] for p in entries),
            "weights_sha256": digest(entries)}


def evaluation_model_sha256(path: str | Path) -> str:
    """Use the frozen evaluator's identity so deployment and accuracy cannot mix."""
    from .evaluate import model_identity
    return digest(model_identity(str(rooted(path))))


def benchmark_inputs(tokenizer: Any, count: int = 512) -> tuple[list[list[int]], dict]:
    from .data import load_items
    from .prompts import build_messages
    rows = load_items(task="gsm8k", split="train")[:count]
    if len(rows) != count or len({r["id"] for r in rows}) != count:
        raise ValueError(f"Need {count} unique training prompts")
    texts = [tokenizer.apply_chat_template(build_messages(r["question"]),
             tokenize=False, add_generation_prompt=True) for r in rows]
    tokens = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    ids = [r["id"] for r in rows]
    return tokens, {"prompt_ids": ids, "prompt_ids_sha256": digest(ids),
                    "prompt_tokens_sha256": digest(tokens),
                    "prompt_lengths": [len(t) for t in tokens]}


def validate_config(config: dict) -> None:
    required = {"num_prompts": 512, "output_tokens": 256, "warmup_runs": 1, "repeats": 3}
    for key, expected in required.items():
        if config[key] != expected:
            raise ValueError(f"Frozen serving comparison requires {key}={expected}")
    if config["hf_batch_sizes"] != [1, 8, 16, 32]:
        raise ValueError("HF baseline must test batch sizes 1, 8, 16, 32")


def measured_trial(seconds: float, token_counts: list[int], count: int = 512,
                   output_tokens: int = 256) -> dict:
    if len(token_counts) != count or any(n != output_tokens for n in token_counts):
        raise ValueError("A timed trial did not produce the required fixed-length outputs")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("A timed trial must have positive finite elapsed time")
    return {"seconds": seconds, "num_outputs": len(token_counts),
            "total_output_tokens": sum(token_counts),
            "min_output_tokens": min(token_counts), "max_output_tokens": max(token_counts),
            "output_tokens_per_second": sum(token_counts) / seconds}


def summarize_trials(trials: list[dict]) -> dict:
    if len(trials) != 3:
        raise ValueError("Serving result requires three measured trials")
    for trial in trials:
        if (not math.isfinite(float(trial["seconds"])) or float(trial["seconds"]) <= 0
                or trial.get("num_outputs") != 512 or trial.get("min_output_tokens") != 256
                or trial.get("max_output_tokens") != 256):
            raise ValueError("Serving trial has invalid time or actual output counts")
    median = statistics.median(float(t["seconds"]) for t in trials)
    totals = {int(t["total_output_tokens"]) for t in trials}
    if totals != {512 * 256}:
        raise ValueError("Trials do not share exactly 512 x 256 output tokens")
    return {"trials": trials, "median_seconds": median,
            "output_tokens_per_second": 512 * 256 / median}


def offline_speedup(hf: dict | None, vllm: dict | None) -> float | None:
    if not hf or not vllm or hf.get("status") != "measured" or vllm.get("status") != "measured":
        return None
    for key in ("model_weights_sha256", "prompt_ids_sha256", "prompt_tokens_sha256",
                "num_prompts", "output_tokens", "protocol_sha256", "model_sha256"):
        if not hf.get(key) or hf[key] != vllm.get(key):
            raise ValueError(f"HF/vLLM comparison does not share {key}")
    batches = hf["batch_results"]
    if [r["batch_size"] for r in batches] != [1, 8, 16, 32]:
        raise ValueError("HF comparison is missing a required batch-size attempt")
    good = [r for r in batches if r["status"] == "measured"]
    if any(r["status"] not in {"measured", "oom"} for r in batches) or not good:
        raise ValueError("HF comparison includes invalid or unmeasured batch attempts")
    best = max(summarize_trials(r["trials"])["output_tokens_per_second"] for r in good)
    v_rate = summarize_trials(vllm["trials"])["output_tokens_per_second"]
    return v_rate / best
