"""Shared training setup; imports GPU dependencies only when training starts."""
from __future__ import annotations

import copy
import importlib.metadata
import json
import re
import time
from pathlib import Path

from .common import ROOT, load_config, write_json
from .data import load_gsm8k_train, load_split
from .protocol import require_memory_decision, verify_frozen


def project_path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def assert_frozen() -> None:
    verify_frozen()


def memory_decision() -> dict:
    decision = require_memory_decision()
    if decision.get("status") != "selected" or not decision.get("decided_before_evaluation"):
        raise RuntimeError("A pre-evaluation R12 memory decision is required before training.")
    if not re.fullmatch(r"[0-9a-f]{40}", decision.get("revision", "")):
        raise RuntimeError("The selected base model must have an immutable Hub revision.")
    return decision


def resolve_model(config: dict, decision: dict) -> tuple[str, str | None]:
    source = config["model"]
    if source == "base":
        return decision["model_id"], decision["revision"]
    path = project_path("checkpoints/sft_merged" if source == "sft_merged" else source)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Merged starting model is missing: {path}")
    lineage_path = path / "merge_manifest.json"
    if not lineage_path.is_file():
        raise RuntimeError(f"Merged starting model has no lineage manifest: {path}")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if lineage.get("selected_model_id") != decision["model_id"] or lineage.get("selected_revision") != decision["revision"]:
        raise RuntimeError("Merged model lineage differs from the frozen R12 model decision.")
    return str(path), None


def pinned_local_model(model: str, revision: str | None) -> str:
    """TRL's colocated vLLM loader omits revision; a local snapshot closes that gap."""
    if revision is None:
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=revision,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer.*"])


def validate_lora(values: dict) -> None:
    required = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                "target_modules": "all-linear", "task_type": "CAUSAL_LM"}
    for key, expected in required.items():
        if values.get(key) != expected:
            raise ValueError(f"Protocol requires LoRA {key}={expected!r}")


def prepare_trainer_kwargs(config: dict, revision: str | None, *, smoke: bool = False) -> dict:
    values = copy.deepcopy(config["trainer"])
    values["output_dir"] = str(project_path(config["output_dir"]))
    model_kwargs = values.setdefault("model_init_kwargs", {})
    if revision is not None:
        model_kwargs["revision"] = revision
    else:
        model_kwargs.pop("revision", None)
    if values.get("seed") != 42 or values.get("data_seed") != 42:
        raise ValueError("Training and data seeds must both be 42.")
    if not values.get("bf16") or not values.get("gradient_checkpointing"):
        raise ValueError("The 12 GB protocol requires bf16 and gradient checkpointing.")
    if smoke:
        values.update(max_steps=20, save_strategy="no", eval_strategy="no",
                      log_completions=False, logging_steps=1)
    return values


def training_dataset():
    source = load_gsm8k_train()
    ids = load_split()["train"]
    if len(ids) != 6973:
        raise ValueError("Expected all 6,973 training problems from the frozen split.")
    return source.select(ids)


def package_versions() -> dict:
    result = {}
    for package in ("torch", "transformers", "trl", "vllm", "peft", "datasets", "accelerate"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def write_run_manifest(config: dict, args, model: str, revision: str | None, decision: dict) -> None:
    write_json(project_path(config["log_dir"]) / "run_manifest.json", {
        "stage": config["stage"], "model": model, "revision": revision,
        "memory_decision": decision, "config": config, "resolved_trainer": args.to_dict(),
        "package_versions": package_versions(), "train_count": 6973, "dev_used_in_training": False,
    })


def logging_callbacks(log_dir: str):
    """Persist each optimizer step duration and every TRL metric as JSONL."""
    import torch
    from transformers import TrainerCallback

    directory = project_path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)

    class MetricsCallback(TrainerCallback):
        def __init__(self):
            self.times: list[float] = []
            self._start = None

        def on_train_begin(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_step_begin(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self._start is not None:
                elapsed = time.perf_counter() - self._start
                self.times.append(elapsed)
                with (directory / "step_times.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": state.global_step, "seconds": elapsed}) + "\n")

        def on_log(self, args, state, control, logs=None, **kwargs):
            row = {"step": state.global_step, "epoch": state.epoch, **(logs or {})}
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")

        def measurements(self) -> dict:
            import statistics
            return {
                "steps_completed": len(self.times), "step_seconds": self.times,
                "median_step_seconds": statistics.median(self.times) if self.times else None,
                "mean_step_seconds": statistics.mean(self.times) if self.times else None,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
            }

    return MetricsCallback()


def ensure_new_or_resumed(config: dict, resume: str | None) -> None:
    output = project_path(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Training output already contains files: {output}; use --resume explicitly.")
    if resume and not project_path(resume).is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")


def assert_grpo_pair(main_path="configs/grpo.yaml", ablation_path="configs/grpo_no_sft.yaml") -> None:
    main = copy.deepcopy(load_config(project_path(main_path)))
    ablation = copy.deepcopy(load_config(project_path(ablation_path)))
    if main.pop("model") != "sft_merged" or ablation.pop("model") != "base":
        raise ValueError("GRPO and its ablation must start from SFT and the base model respectively.")
    for config in (main, ablation):
        config.pop("output_dir")
        config.pop("log_dir")
        config["selection"].pop("result_dir")
        config["selection"].pop("output_path")
    if main != ablation:
        raise ValueError("Ablation must differ only in starting model and output paths.")
