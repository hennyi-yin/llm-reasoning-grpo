import pytest

from scripts.analyze_saved_results import describe_format_strata


def row(boxed, correct, question="q"):
    return {"has_boxed": str(boxed), "correct": str(correct), "question_sha256": question}


def test_format_strata_keep_losses_and_use_full_paired_denominator():
    left = {"1": row(False, False), "2": row(True, True),
            "3": row(True, False), "4": row(False, False)}
    right = {"4": row(False, False), "3": row(True, True),
             "2": row(True, False), "1": row(True, True)}
    result = describe_format_strata(left, right)
    strata = {(r["a_has_boxed"], r["b_has_boxed"]): r for r in result["strata"]}
    assert result["net_correct"] == 1
    assert result["difference_pp"] == 25
    assert strata[False, True]["difference_pp_full_test_denominator"] == 25
    assert strata[True, True]["gain"] == strata[True, True]["loss"] == 1
    assert strata[True, False]["n"] == 0
    assert sum(r["net_correct"] for r in result["strata"]) == result["net_correct"]


def test_format_strata_reject_equal_size_different_problem_ids():
    with pytest.raises(ValueError, match="problem IDs"):
        describe_format_strata({"1": row(True, True)}, {"2": row(True, True)})


def test_format_strata_reject_changed_question_under_same_id():
    with pytest.raises(ValueError, match="different questions"):
        describe_format_strata({"1": row(True, True)}, {"1": row(True, True, "other")})
