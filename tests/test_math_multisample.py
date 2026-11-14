import pytest

from src.math_multisample import pass_k, summarize, vote_index, vote_key


def test_pass_k_uses_all_subsets():
    eight = [True] + [False] * 7
    assert [pass_k(eight, k) for k in (1, 2, 4, 8)] == [0.125, 0.25, 0.5, 1]
    assert pass_k([False] * 8, 8) == 0
    assert pass_k([True] * 8, 1) == 1
    with pytest.raises(ValueError, match="sampling budget"):
        pass_k(eight, 9)


def test_reference_blind_votes_normalize_fractions_and_abstain():
    keys = [
        vote_key(r"Therefore \boxed{\frac{2}{4}}.", False),
        vote_key(r"Therefore \boxed{2}.", False),
        vote_key("Final answer: 0.5", False),
        vote_key("Final answer: 0.5 or 2", False),
        vote_key("Final answer: 2", True),
    ]
    assert keys == ["number:1/2", "number:2", "number:1/2", None, None]
    assert vote_index(keys) == 0
    assert vote_index([None, "b", "a", "a", "b"]) == 1
    assert vote_index([None, None]) is None


def test_paired_coverage_is_distinct_from_voted_correctness():
    def sample(key, correct):
        return dict(vote_key=key, correct=correct, strict=correct,
                    source="boxed", truncated=False, n_tokens=20)

    right, wrong = sample("right", True), sample("wrong", False)
    rows = [
        dict(base=[right] + [wrong] * 7, grpo=[right] * 5 + [wrong] * 3),
        dict(base=[wrong] * 8, grpo=[wrong] * 7 + [right]),
    ]
    config = dict(questions=2, sampling={"n": 8},
                  analysis={"k": [1, 2, 4, 8], "bootstrap_seed": 0, "bootstrap_replicates": 20})
    result = summarize(rows, config)
    assert result["coverage"]["pass@8"]["gains"] == 1
    assert result["coverage"]["pass@8"]["losses"] == 0
    assert result["arms"]["base"]["covered_but_not_selected"] == 1
    assert result["arms"]["grpo"]["covered_but_not_selected"] == 1
    assert result["vote_at_8"]["grpo_pct"] == 50

    rows[0]["base"].pop()
    with pytest.raises(ValueError, match="Incomplete paired sampling"):
        summarize(rows, config)
