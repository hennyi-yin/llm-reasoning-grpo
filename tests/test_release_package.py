import json
import tarfile

import pytest

from scripts.package_release import hash_file, package


def fixture(root):
    (root / "results").mkdir()
    (root / "results/summary.json").write_text(json.dumps({"status": "measured", "resume_values": {"A": 1}}))
    (root / "configs").mkdir()
    (root / "configs/memory_decision.json").write_text(json.dumps({"model_id": "Qwen/test", "revision": "a" * 40}))
    (root / "licenses").mkdir()
    (root / "licenses/Qwen--test-LICENSE").write_text("fixture upstream license")
    for stage in ("sft", "grpo", "ablation"):
        directory = root / "checkpoints" / stage / "checkpoint-50"
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text("{}")
        (directory / "adapter_model.safetensors").write_bytes(b"selected adapter")
        (directory / "optimizer.pt").write_bytes(b"must not be packaged")
        files = [directory / "adapter_config.json", directory / "adapter_model.safetensors"]
        selected = {"checkpoint": directory.relative_to(root).as_posix(),
                    "artifact_sha256": {path.name: hash_file(path) for path in files}}
        (root / f"results/selected_{stage}.json").write_text(json.dumps(selected))
    for relative in ("checkpoints/sft_merged", "models/final-bf16", "models/ablation-bf16", "models/final-w4a16"):
        directory = root / relative
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(b"merged weights")


def test_all_selected_and_merged_models_packaged_without_optimizer(tmp_path):
    fixture(tmp_path)
    output = tmp_path / "release_assets"
    manifest = package(tmp_path, output)
    assert len(manifest["artifacts"]) == 7
    assert manifest["uploaded"] is False
    with tarfile.open(output / "grpo-adapter.tar.gz") as archive:
        assert set(archive.getnames()) == {"adapter_config.json", "adapter_model.safetensors", "UPSTREAM_MODEL_LICENSE"}
    assert package(tmp_path, output) == manifest


def test_changed_release_source_is_rejected_without_overwriting_archive(tmp_path):
    fixture(tmp_path)
    output = tmp_path / "release_assets"
    package(tmp_path, output)
    archive = output / "final-w4a16.tar.gz"
    original = archive.read_bytes()
    (tmp_path / "models/final-w4a16/model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="source artifact changed"):
        package(tmp_path, output)
    assert archive.read_bytes() == original


def test_changed_selected_adapter_is_rejected_before_packaging(tmp_path):
    fixture(tmp_path)
    (tmp_path / "checkpoints/grpo/checkpoint-50/adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="adapter changed"):
        package(tmp_path, tmp_path / "release_assets")
    assert not (tmp_path / "release_assets").exists()


def test_pending_results_cannot_be_packaged_as_complete(tmp_path):
    fixture(tmp_path)
    (tmp_path / "results/summary.json").write_text('{"status":"pending_gpu_execution"}')
    with pytest.raises(RuntimeError, match="Finish all measured"):
        package(tmp_path, tmp_path / "release_assets")
