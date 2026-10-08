from pathlib import Path
from subprocess import CompletedProcess

import pytest

import scripts.check_quant_gpu as checker


def test_fixture_is_built_and_checked_across_both_interpreters(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "__file__", str(tmp_path / "scripts/check_quant_gpu.py"))
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "model.safetensors").write_bytes(b"fixture")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = f"CPU_COMPAT_CHECKPOINT={fixture}\n" if command[-1] == "produce" else "verified\n"
        return CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(checker.subprocess, "run", run)
    assert checker.prepare_synthetic_checkpoint("quant-python") == fixture
    assert calls[0][0] == "quant-python"
    assert calls[1][0] == checker.sys.executable
    assert calls[1][-3:] == ["consume", "--checkpoint", str(fixture)]
    assert "verified" in (tmp_path / "logs/quantization-compatibility-cpu.log").read_text()


@pytest.mark.parametrize("returncode, stdout, expected", [
    (1, "", "producer failed"),
    (0, "CPU_COMPAT_CHECKPOINT=/missing/fixture\n", "complete checkpoint"),
])
def test_gpu_does_not_start_after_invalid_fixture(tmp_path, monkeypatch, returncode, stdout, expected):
    monkeypatch.setattr(checker, "__file__", str(tmp_path / "scripts/check_quant_gpu.py"))
    monkeypatch.setattr(checker.subprocess, "run", lambda command, **kwargs:
                        CompletedProcess(command, returncode, stdout, "failure diagnostics"))
    with pytest.raises(RuntimeError, match=expected):
        checker.prepare_synthetic_checkpoint("quant-python")
    assert "failure diagnostics" in (tmp_path / "logs/quantization-compatibility-cpu.log").read_text()
