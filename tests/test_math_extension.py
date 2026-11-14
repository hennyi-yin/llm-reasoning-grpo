from random import Random

from src.math_independent import select_subject, summarize
from src.math_self_sft import summarize as summarize_self


def answer(correct, source="boxed", tokens=400):
    return dict(correct=correct, previous=correct, strict=correct, source=source,
                n_tokens=tokens, truncated=False)


def test_stratified_selection_excludes_train_and_duplicate_test_problems():
    rows = [{"id": str(i), "question_sha256": str(i), "level": i // 10} for i in range(20)]
    blocked = {"0", "19"}
    picked = select_subject(rows + [rows[3]], blocked, 10, Random(3))
    assert len({x["id"] for x in picked}) == 10
    assert {x["id"] for x in picked}.isdisjoint({"0", "19"})
    assert sum(x["level"] == 0 for x in picked) == 5


def test_serving_gate_requires_accuracy_preservation_and_shorter_outputs():
    cfg = dict(noninferiority_margin_pp=3, worthwhile_gain_pp=2, max_unresolved_fraction=.25,
               max_extra_unresolved_pp=3, min_token_reduction_fraction=.03)
    equal = [dict(base=answer(True), candidate=answer(True, tokens=300))] * 100
    result = summarize(equal, cfg)
    assert result["serving_gate"] and not result["improved"]
    assert result["default"] == "base"
    equal[:20] = [dict(base=answer(True), candidate=answer(False, tokens=300))] * 20
    assert not summarize(equal, cfg)["serving_gate"]


def test_self_generated_gain_against_only_one_control_does_not_trigger_more_training():
    gate = dict(min_gain_pp=2, max_unresolved_fraction=.25, max_extra_unresolved_pp=3)
    rows = [dict(base=answer(True), reference=answer(False), self=answer(True))] * 100
    result = summarize_self(rows, gate)
    assert result["decision"] == "stop"
    assert not result["grpo_started"]
