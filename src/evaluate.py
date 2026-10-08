"""Frozen greedy evaluation with a durable, resumable, once-only final ledger.

Final evaluation claims a model/task before accessing test data. Raw generations
are durable before judging. A crash during an uncommitted generation is blocked:
silently retrying it would violate the once-only test protocol.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone

from .common import ROOT, load_config, resolve, sha256_file
from .extract import last_boxed, numeric_equal
from .prompts import build_messages

EXPECTED_COUNTS = {"gsm8k": 1319, "math500": 500}

FIELDS = ["model", "task", "id", "correct", "has_boxed", "n_tokens", "truncated",
          "split", "question_sha256", "model_sha256", "protocol_sha256",
          "extracted_answer", "finish_reason", "raw_output_path", "raw_output_sha256"]


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _durable_json(path: Path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    if os.name != "nt":
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


@contextmanager
def _lock(path: Path):
    """OS advisory lock releases on process death; the durable claim does not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("This evaluation is already running") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This evaluation is already running") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def judge(task: str, gold, text: str) -> bool:
    answer = last_boxed(text)
    if answer is None or not answer.strip():
        return False
    if task == "gsm8k":
        return numeric_equal(answer, gold)
    if task != "math500":
        raise ValueError(f"Unknown task: {task}")
    from math_verify import LatexExtractionConfig, parse, verify
    # Never pass the reasoning to Math-Verify: a correct earlier expression must
    # not rescue an incorrect or malformed final boxed answer.
    config = [LatexExtractionConfig(boxed_match_priority=0)]
    target = parse(f"\\boxed{{{gold}}}", extraction_config=config,
                   extraction_mode="first_match")
    prediction = parse(f"\\boxed{{{answer}}}", extraction_config=config,
                       extraction_mode="first_match")
    return bool(target and prediction and verify(target, prediction))


def validate_config(config: dict):
    required = {"seed": 0, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
                "n": 1, "dtype": "bfloat16"}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Evaluation must use the frozen greedy seed-0 protocol")
    if config.get("max_tokens") != {"gsm8k": 1024, "math500": 2048}:
        raise ValueError("Frozen token caps must be GSM8K 1024 / MATH-500 2048")
    if config.get("numeric_tolerance") != 0.000001:
        raise ValueError("Frozen numeric tolerance is 1e-6")


def model_identity(model_path: str, revision: str | None = None,
                   adapter_path: str | None = None) -> dict:
    """Hash actual local artifacts or a pinned remote revision, including LoRA."""
    local = resolve(model_path)
    identity = {"path": str(local.resolve()) if local.is_dir() else model_path,
                "revision": revision, "files": {}}
    if local.is_dir():
        files = sorted(p for p in local.rglob("*") if p.is_file() and
                       (p.suffix in {".safetensors", ".bin", ".json", ".model", ".txt"})
                       and not any(part.startswith(".") for part in p.relative_to(local).parts))
        if not files or not (local / "config.json").exists():
            raise ValueError(f"Invalid local model artifact: {local}")
        identity["files"] = {str(p.relative_to(local)): sha256_file(p) for p in files}
    elif not revision or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("A remote evaluation model must use an exact 40-character revision")
    if adapter_path:
        adapter = resolve(adapter_path)
        files = sorted(p for p in adapter.iterdir() if p.is_file() and p.suffix in {".json", ".bin", ".safetensors"})
        if not (adapter / "adapter_config.json").is_file() or not files:
            raise ValueError(f"Invalid LoRA adapter: {adapter}")
        identity["adapter"] = {"path": str(adapter.resolve()),
                               "files": {p.name: sha256_file(p) for p in files}}
    return identity


def _make_generator(model_path, revision, adapter_path, config, task):
    from vllm import LLM, SamplingParams
    local = resolve(model_path)
    kwargs = {"model": str(local) if local.is_dir() else model_path,
              "dtype": config["dtype"], "seed": config["seed"],
              "gpu_memory_utilization": config["gpu_memory_utilization"],
              "max_model_len": config["max_model_len"],
              "enable_lora": bool(adapter_path)}
    if revision and not local.is_dir():
        kwargs["revision"] = revision
        kwargs["tokenizer_revision"] = revision
    if adapter_path:
        kwargs["max_lora_rank"] = 16
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, n=1,
                            max_tokens=config["max_tokens"][task], seed=0)
    lora_request = None
    if adapter_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("evaluation", 1, str(resolve(adapter_path)))

    def generate(batch):
        outputs = llm.chat([build_messages(item["question"]) for item in batch], params,
                           lora_request=lora_request)
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM returned an incorrect number of outputs")
        result = []
        for output in outputs:
            if len(output.outputs) != 1:
                raise RuntimeError("Greedy evaluation requires exactly one output per question")
            completion = output.outputs[0]
            result.append({"text": completion.text, "token_ids": list(completion.token_ids),
                           "finish_reason": completion.finish_reason,
                           "stop_reason": completion.stop_reason})
        return result
    return generate


class EvaluationStore:
    """Immutable generation claims and raw outputs for one exact model/task."""

    def __init__(self, directory: Path, identity: dict):
        self.directory = directory
        self.manifest_path = directory / "manifest.json"
        self.identity = identity
        directory.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            _durable_json(self.manifest_path, {"schema": 1, "identity": identity,
                          "started_at": datetime.now(timezone.utc).isoformat(),
                          "status": "started", "items": None, "in_flight": [],
                          "raw_sha256": {}}, exclusive=True)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("identity") != identity:
            raise RuntimeError("Evaluation was already claimed with different model/protocol identity")

    def _save(self):
        _durable_json(self.manifest_path, self.manifest)

    def raw_path(self, problem_id):
        return self.directory / "raw" / (_text_digest(str(problem_id)) + ".json")

    def initialize(self, items: list[dict]):
        ids = [str(item["id"]) for item in items]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Evaluation requires nonempty unique problem IDs")
        description = [{"id": str(item["id"]), "question_sha256": _text_digest(item["question"]),
                        "gold_sha256": _digest(item["gold"])} for item in items]
        if self.manifest["items"] is not None and self.manifest["items"] != description:
            raise RuntimeError("Dataset identity differs from the original evaluation claim")
        self.manifest["items"] = description
        self._save()

    def recover(self):
        inflight = self.manifest["in_flight"]
        if any(not self.raw_path(problem_id).exists() for problem_id in inflight):
            raise RuntimeError("An interrupted generation has no committed raw output. Test reruns are prohibited; retain this ledger and investigate the original process/logs.")
        if inflight:
            # A process may die after saving raw files and before saving their
            # hashes. Only those still claimed in flight can finish this commit.
            for problem_id in inflight:
                digest = sha256_file(self.raw_path(problem_id))
                known = self.manifest["raw_sha256"].get(problem_id)
                if known and digest != known:
                    raise RuntimeError("Committed raw generation has changed")
                self.manifest["raw_sha256"][problem_id] = digest
            self.manifest["in_flight"] = []
            self._save()

    def complete_raw(self) -> bool:
        items = self.manifest.get("items")
        return bool(items) and all(self.raw_path(item["id"]).exists() for item in items)

    def generate(self, items: list[dict], factory, batch_size: int):
        self.initialize(items)
        self.recover()
        pending = [item for item in items if not self.raw_path(item["id"]).exists()]
        if pending:
            generate = factory()
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                self.manifest["in_flight"] = [str(item["id"]) for item in batch]
                self._save()  # Durable claim precedes generation.
                outputs = generate(batch)
                if len(outputs) != len(batch):
                    raise RuntimeError("Generator returned an incorrect number of outputs")
                for item, output in zip(batch, outputs):
                    if not isinstance(output.get("text"), str) or not isinstance(output.get("token_ids"), list):
                        raise ValueError("Invalid generation output")
                    _durable_json(self.raw_path(item["id"]),
                                  {"schema": 1, "identity": self.identity,
                                   "problem": item, "generation": output}, exclusive=True)
                    self.manifest["raw_sha256"][str(item["id"])] = sha256_file(self.raw_path(item["id"]))
                    self._save()
                self.manifest["in_flight"] = []
                self._save()
        return self.rows()

    def rows(self) -> list[dict]:
        self.recover()
        if not self.complete_raw():
            raise RuntimeError("Evaluation is incomplete")
        rows = []
        for item in self.manifest["items"]:
            path = self.raw_path(item["id"])
            digest = sha256_file(path)
            if self.manifest["raw_sha256"].get(item["id"]) != digest:
                raise RuntimeError("Committed raw generation has changed or lacks a durable hash")
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("identity") != self.identity:
                raise RuntimeError("Raw output identity differs from the evaluation claim")
            problem, output = raw["problem"], raw["generation"]
            if str(problem["id"]) != item["id"] or _text_digest(problem["question"]) != item["question_sha256"] or _digest(problem["gold"]) != item["gold_sha256"]:
                raise RuntimeError("Raw problem identity differs from the dataset claim")
            text = output["text"]
            try:
                raw_path = str(path.relative_to(ROOT))
            except ValueError:
                raw_path = str(path)
            rows.append({"model": self.identity["model"], "task": self.identity["task"],
                         "id": item["id"], "correct": judge(self.identity["task"], problem["gold"], text),
                         "has_boxed": last_boxed(text) is not None,
                         "n_tokens": len(output["token_ids"]),
                         "truncated": output.get("finish_reason") == "length",
                         "split": self.identity["split"], "question_sha256": item["question_sha256"],
                         "model_sha256": self.identity["model_sha256"],
                         "protocol_sha256": self.identity["protocol_sha256"],
                         "extracted_answer": last_boxed(text), "finish_reason": output.get("finish_reason"),
                         "raw_output_path": raw_path, "raw_output_sha256": digest})
        self.manifest["status"] = "complete"
        self.manifest["n"] = len(rows)
        self._save()
        return rows


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _refresh_final_csv():
    ledger = ROOT / "results/final_eval"
    with _lock(ledger / "aggregate.lock"):
        rows = []
        for path in sorted(ledger.glob("*/manifest.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["status"] == "complete":
                # Read committed outputs only; this never reloads a dataset or model.
                store = EvaluationStore(path.parent, manifest["identity"])
                rows.extend(store.rows())
        _write_csv(ROOT / "results/per_problem.csv", rows)


def _identity(label, model_path, revision, adapter_path, task, split):
    return {"model": label, "task": task, "split": split,
            "model_sha256": _digest(model_identity(model_path, revision, adapter_path)),
            "protocol_sha256": sha256_file("docs/frozen_manifest.json")}


def run_dev(model_path, *, base_model=None, revision=None, adapter_path=None,
            config_path="configs/eval.yaml", output_path=None) -> list[dict]:
    from .protocol import require_memory_decision
    from .data import load_items
    require_memory_decision()
    config = load_config(config_path)
    validate_config(config)
    # No alternate config can alter the evaluation while preserving its provenance.
    if sha256_file(config_path) != sha256_file("configs/eval.yaml"):
        raise ValueError("Evaluation config differs from the frozen config")
    actual_model = base_model or model_path
    label = str(adapter_path or model_path)
    identity = _identity(label, actual_model, revision, adapter_path, "gsm8k", "dev")
    items = load_items("gsm8k", "dev")
    # Dev can be evaluated repeatedly during model selection. Separate invocations
    # get separate artifacts; only test evaluation uses the global once-only ledger.
    directory = (resolve(output_path).parent / (resolve(output_path).stem + "_raw")) if output_path else Path(tempfile.mkdtemp(prefix="dev-eval-", dir=ROOT / "results"))
    with _lock(directory / "active.lock"):
        store = EvaluationStore(directory, identity)
        rows = store.generate(items, lambda: _make_generator(actual_model, revision, adapter_path, config, "gsm8k"), config["batch_size"])
    if output_path:
        path = resolve(output_path)
        if path.suffix == ".csv":
            _write_csv(path, rows)
        else:
            _durable_json(path, {"identity": identity, "accuracy": sum(r["correct"] for r in rows) / len(rows), "rows": rows})
    return rows


def run_final(model_path, *, model_label, task, revision=None,
              config_path="configs/eval.yaml") -> list[dict]:
    from .protocol import require_final_eligibility
    from .data import load_items
    registry = require_final_eligibility()  # Always before any test data access.
    config = load_config(config_path)
    validate_config(config)
    if sha256_file(config_path) != sha256_file("configs/eval.yaml"):
        raise ValueError("Evaluation config differs from the frozen config")
    if task not in {"gsm8k", "math500"} or model_label not in registry["models"]:
        raise ValueError("Unknown final model label/task")
    spec = registry["models"][model_label]
    requested = model_identity(model_path, revision)
    registered = model_identity(spec["path"], spec.get("revision"))
    if requested != registered:
        raise ValueError("Requested model artifact differs from the locked final registry")
    identity = _identity(model_label, model_path, revision, None, task, "test")
    identity["registry_sha256"] = _digest(registry)
    directory = ROOT / "results/final_eval" / f"{model_label}--{task}"
    with _lock(directory / "active.lock"):
        store = EvaluationStore(directory, identity)  # Durable start before test loading.
        store.recover()
        if store.complete_raw():
            if len(store.manifest["items"]) != EXPECTED_COUNTS[task]:
                raise RuntimeError("Completed final ledger has an incorrect problem count")
            rows = store.rows()  # Resume or return exact completed outputs, no generation.
        else:
            items = load_items(task, "test")
            expected = EXPECTED_COUNTS[task]
            if len(items) != expected:
                raise ValueError(f"Final evaluation requires exactly {expected} problems")
            rows = store.generate(items, lambda: _make_generator(model_path, revision, None, config, task), config["batch_size"])
    _refresh_final_csv()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Pinned repo ID or local merged model")
    parser.add_argument("--model-label", "--label", dest="model_label", default=None,
                        help="Final registry label: base, sft, grpo, ablation, quantized")
    parser.add_argument("--revision", help="Exact remote model commit")
    parser.add_argument("--adapter", help="LoRA adapter; dev selection only")
    parser.add_argument("--task", choices=["gsm8k", "math500"], default="gsm8k")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--output-dir", help="Dev artifact directory (test always uses results/)")
    args = parser.parse_args()
    if args.split == "test":
        if not args.model_label or args.adapter or args.output_dir:
            parser.error("test requires --model-label, a merged model, and the global results directory")
        rows = run_final(args.model, model_label=args.model_label, task=args.task,
                         revision=args.revision, config_path=args.config)
    else:
        if args.task != "gsm8k":
            parser.error("only GSM8K has a dev split")
        directory = resolve(args.output_dir or "results/dev")
        label = args.model_label or Path(args.adapter or args.model).name
        safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        rows = run_dev(args.model, revision=args.revision, adapter_path=args.adapter,
                       config_path=args.config, output_path=directory / f"{safe_label}.json")
    print(json.dumps({"n": len(rows), "accuracy": sum(row["correct"] for row in rows) / len(rows),
                      "boxed_rate": sum(row["has_boxed"] for row in rows) / len(rows)}))


if __name__ == "__main__":
    main()
