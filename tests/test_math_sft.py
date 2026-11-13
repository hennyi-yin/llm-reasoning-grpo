import pytest

from src.math_sft import fit_ids, paired, score_v2, summarize


def test_revised_judge_accepts_only_supported_final_forms():
    assert score_v2("60", "The tree is 480 inches.\n\nSo, Andrea is 60 inches tall.")[2]
    assert not score_v2("60", "So, Andrea is 60 inches tall.", truncated=True)[2]
    assert not score_v2("60", "We tried 60 inches; no conclusion.")[2]
    assert score_v2(r"\sin^2 t", r"Final answer: \boxed{\sin^2 t}")[2]
    assert not score_v2("7", "Final answer: 7 or 8")[2]


def test_normalized_heldout_overlap_is_rejected():
    with pytest.raises(ValueError, match="held-out"):
        fit_ids([{"id": "different", "question_sha256": "same"}], {"same"}, 1)


def test_paired_comparison_counts_opposite_transitions():
    result = paired([True, False, False, False], [False, True, True, False])
    assert (result["left"], result["right"], result["gains"], result["losses"], result["delta_pp"]) == (1, 2, 2, 1, 25)


def test_gate_requires_worked_gain_over_base_and_answer_only():
    def rows(values):
        return [dict(correct=x, previous=x, strict=x, truncated=False, source="boxed", n_tokens=10)
                for x in values]

    gate = dict(min_gain_vs_base_pp=2, min_gain_vs_short_pp=2, max_unresolved_fraction=0.25,
                max_extra_unresolved_pp=3)
    base = rows([True, False, False, False])
    worked = rows([True, True, True, False])
    assert summarize("dev", dict(base=base, worked=worked, short=worked), gate)["decision"] == "stop"
    assert summarize("dev", dict(base=base, worked=worked, short=base), gate)["decision"] == "confirm"
