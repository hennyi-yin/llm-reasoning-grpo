import pytest
from src.stats import bootstrap_ci, mcnemar_exact, paired_comparison, summarize_rows


def rows(model, values, order=None):
    order = list(range(len(values))) if order is None else order
    return [{"model": model, "task": "gsm8k", "id": f"test-{i}", "correct": values[i],
             "question_sha256": f"question-{i}", "protocol_sha256": "frozen",
             "model_sha256": model, "split": "test", "has_boxed": True,
             "n_tokens": 3, "truncated": False} for i in order]


def test_bootstrap_reproducible_and_extremes():
    assert bootstrap_ci([0] * 8) == [0., 0.]
    assert bootstrap_ci([1] * 8) == [1., 1.]
    assert bootstrap_ci([0, 0, 1, 1]) == bootstrap_ci([0, 0, 1, 1])


@pytest.mark.parametrize("value", [[], [0, 2], [[0, 1]]])
def test_bootstrap_rejects_bad_observations(value):
    with pytest.raises(ValueError):
        bootstrap_ci(value)


def test_mcnemar_known_exact_values():
    assert mcnemar_exact([0] * 6, [1] * 6) == .03125
    assert mcnemar_exact([0, 1], [0, 1]) == 1
    with pytest.raises(ValueError):
        mcnemar_exact([0], [0, 1])


def test_pair_alignment_and_difference_direction():
    result = paired_comparison(rows("a", [0, 0, 1]), rows("b", [1, 0, 1], [2, 0, 1]))
    assert result["n01"] == 1 and result["n10"] == 0
    assert result["difference_pp"] == pytest.approx(100 / 3)


@pytest.mark.parametrize("field", ["question_sha256", "protocol_sha256", "split", "id"])
def test_pair_identity_mismatch(field):
    a, b = rows("a", [0, 1]), rows("b", [0, 1])
    b[0][field] = "different"
    with pytest.raises(ValueError):
        paired_comparison(a, b)


def test_duplicate_identity_rejected():
    a = rows("a", [0, 1])
    with pytest.raises(ValueError):
        paired_comparison(a + a[:1], rows("b", [1, 1]))
    with pytest.raises(ValueError):
        summarize_rows(a + a[:1])


def test_summary_pending_and_partial_are_not_fabricated():
    assert summarize_rows([])["status"] == "incomplete"
    summary = summarize_rows(rows("base", [1, 0]) + rows("sft", [1, 1]))
    assert summary["status"] == "incomplete"
    assert summary["comparisons"][0]["difference_pp"] == 50
