import pytest

from src.followup_answers import final_candidate
from src.math_pilot_data import question_key, split_groups


@pytest.mark.parametrize(
    ("text", "answer", "source"),
    [
        (r"An intermediate \boxed{9}. Final answer: 7", "7", "explicit_final"),
        (r"Intermediate \boxed{9}. Final answer: \boxed{7", None, "review"),
        ("Final answer: **7**.", "7", "explicit_final"),
        ("Final answer: 7 dollars.", "7", "explicit_final"),
        ("Final answer: 12/2", None, "review"),
        ("Final answer: 4 or 5", None, "review"),
        (r"First \[2\]. Then \[7\]", "7", "terminal_display"),
        ("The pen costs 3 dollars. Then I would", None, "review"),
        ("Working: 9; 3+4=7", None, "review"),
    ],
)
def test_final_candidate_does_not_rescue_an_intermediate(text, answer, source):
    assert final_candidate(text, numeric=True) == (answer, source)


def test_followup_split_is_disjoint_reproducible_and_balanced():
    subjects = ["algebra", "geometry"]
    data = [{"id": f"{s}-{i}", "subject": s} for s in subjects for i in range(20)]
    cfg = dict(seed=42, dev_count=5, confirmation_count=6, screening_count=7)
    split = split_groups(data, subjects, cfg)
    assert split == split_groups(list(reversed(data)), subjects, cfg)
    assert [len(split[k]) for k in ("dev", "confirmation", "screen")] == [5, 6, 7]
    assert len({x["id"] for items in split.values() for x in items}) == 18
    assert {x["subject"] for x in split["dev"]} == set(subjects)
    assert question_key("  Find \uFF58 + 2") == question_key("find x+2")


def test_math_reward_accepts_equivalent_unboxed_answer_without_rewarding_working():
    pytest.importorskip("math_verify")
    from src.math_pilot import score
    assert score(r"\frac{2}{3}", "Final answer: 2/3")[2]
    assert not score("7", "We computed 7 earlier. Final answer: 8")[2]
    assert not score("7", "Final answer: 7 or 8")[2]
    assert not score("7", "We computed 7 earlier and then")[2]
    assert not score("7", "Final answer: 7", truncated=True)[2]
