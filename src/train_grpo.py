"""GRPO with verifiable rewards; the ablation uses exactly the same trainer."""
from __future__ import annotations

import argparse
import copy

from .common import load_config, write_json
from .extract import gsm8k_gold
from .prompts import build_messages
from .rewards import correctness_reward, format_reward
from .training import (assert_frozen, assert_grpo_pair, ensure_new_or_resumed, logging_callbacks,
                       memory_decision, prepare_trainer_kwargs, project_path, resolve_model,
                       training_dataset, validate_lora, write_run_manifest, pinned_local_model)


def to_rl(row: dict) -> dict:
    return {"prompt": build_messages(row["question"]), "gold": gsm8k_gold(row["answer"])}


def train(config_path: str, resume: str | None = None, *, smoke_candidate: dict | None = None) -> dict:
    assert_frozen()
    config = copy.deepcopy(load_config(project_path(config_path)))
    if config["stage"] != "grpo":
        raise ValueError("Expected a GRPO configuration.")
    smoke = smoke_candidate is not None
    if smoke:
        decision = {"status": "smoke", **smoke_candidate}
        model, revision = smoke_candidate["model_id"], smoke_candidate["revision"]
        config["trainer"].update(smoke_candidate["grpo_overrides"])
    else:
        assert_grpo_pair()
        decision = memory_decision()
        model, revision = resolve_model(config, decision)
        config["trainer"].update(decision["grpo_overrides"])
    ensure_new_or_resumed(config, resume)
    validate_lora(config["lora"])
    required = {"num_generations": 8, "max_completion_length": 512, "temperature": 1.0,
                "learning_rate": 1e-5, "beta": 0.0, "loss_type": "dapo"}
    for key, value in required.items():
        if config["trainer"].get(key) != value:
            raise ValueError(f"Protocol requires GRPO {key}={value}")
    values = prepare_trainer_kwargs(config, revision, smoke=smoke)
    if values["generation_batch_size"] % values["num_generations"]:
        raise ValueError("Generation batch size must divide into complete groups of eight.")
    if values["use_vllm"] and values["use_transformers_continuous_batching"]:
        raise ValueError("Choose exactly one generation engine.")

    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from transformers import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("GRPO requires a CUDA GPU; CPU smoke results cannot choose a 12 GB memory plan.")
    set_seed(42)
    source = training_dataset()
    dataset = source.map(to_rl, remove_columns=source.column_names, desc="Build training-only GRPO data")
    args = GRPOConfig(**values)
    write_run_manifest(config, args, model, revision, decision)
    callback = logging_callbacks(config["log_dir"])
    local_model = pinned_local_model(model, revision)
    trainer = GRPOTrainer(model=local_model, reward_funcs=[correctness_reward, format_reward], args=args,
                          train_dataset=dataset, peft_config=LoraConfig(**config["lora"]), callbacks=[callback])
    trainer.train(resume_from_checkpoint=str(project_path(resume)) if resume else None)
    if not smoke:
        trainer.save_state()
        trainer.save_model(str(project_path(config["output_dir"]) / "final_adapter"))
        trainer.processing_class.save_pretrained(project_path(config["output_dir"]) / "final_adapter")
    report = callback.measurements()
    write_json(project_path(config["log_dir"]) / "training_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--smoke-candidate", help="Internal R12 worker JSON; bypasses memory decision only.")
    args = parser.parse_args()
    candidate = load_config(project_path(args.smoke_candidate)) if args.smoke_candidate else None
    train(args.config, args.resume, smoke_candidate=candidate)


if __name__ == "__main__":
    main()
