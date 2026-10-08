"""Completion-only LoRA SFT on the frozen GSM8K training partition."""
from __future__ import annotations

import argparse
import re

from .common import load_config, write_json
from .prompts import build_messages
from .training import (assert_frozen, ensure_new_or_resumed, logging_callbacks, memory_decision,
                       prepare_trainer_kwargs, project_path, resolve_model, training_dataset,
                       validate_lora, write_run_manifest)


def to_sft(row: dict) -> dict:
    parts = row["answer"].rsplit("####", 1)
    if len(parts) != 2:
        raise ValueError("GSM8K training answer lacks the final #### separator.")
    steps, gold = parts
    steps = re.sub(r"<<[^>]*>>", "", steps).strip()
    reply = f"{steps}\nThe final answer is \\boxed{{{gold.strip()}}}."
    return {"prompt": build_messages(row["question"]),
            "completion": [{"role": "assistant", "content": reply}]}


def train(config_path: str, resume: str | None = None) -> dict:
    assert_frozen()
    decision = memory_decision()
    config = load_config(project_path(config_path))
    if config["stage"] != "sft" or config["model"] != "base":
        raise ValueError("SFT must use the frozen base model.")
    if config["trainer"].get("completion_only_loss") is not True:
        raise ValueError("SFT requires completion_only_loss=True.")
    if config["trainer"].get("num_train_epochs") not in (1, 2):
        raise ValueError("SFT must train one or two epochs.")
    ensure_new_or_resumed(config, resume)
    validate_lora(config["lora"])
    model, revision = resolve_model(config, decision)

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    from transformers import set_seed

    set_seed(42)
    source = training_dataset()
    dataset = source.map(to_sft, remove_columns=source.column_names, desc="Build completion-only SFT data")
    args = SFTConfig(**prepare_trainer_kwargs(config, revision))
    write_run_manifest(config, args, model, revision, decision)
    callback = logging_callbacks(config["log_dir"])
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                         peft_config=LoraConfig(**config["lora"]), callbacks=[callback])
    # The collator's labels must mask every prompt token. Inspect a real batch before updating weights.
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]
    if not (labels == -100).any().item() or not (labels != -100).any().item():
        raise RuntimeError("SFT labels lack either masked prompt tokens or trainable completion tokens.")
    trainer.train(resume_from_checkpoint=str(project_path(resume)) if resume else None)
    trainer.save_state()
    trainer.save_model(str(project_path(config["output_dir"]) / "final_adapter"))
    trainer.processing_class.save_pretrained(project_path(config["output_dir"]) / "final_adapter")
    report = callback.measurements()
    write_json(project_path(config["log_dir"]) / "training_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
