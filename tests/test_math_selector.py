from src.math_multisample import vote_index
from src.math_selector import develop, evidence, rank_index, summarize
from src.math_selector_experiment import question_seed


def sample(key, tokens, correct, *, truncated=False):
    return {"vote_key": key, "n_tokens": tokens, "truncated": truncated,
            "correct": correct, "gold": "should never enter the ranker"}


def test_plurality_prior_and_reference_blind_ranking():
    rows = [
        sample("number:1", 1000, True),
        sample("number:2", 100, False),
        sample("number:2", 110, False),
    ]
    blind = evidence(rows)
    assert {"vote_key", "n_tokens", "truncated"} == set(blind[0])
    assert rank_index(blind, [1, 0, 0, 0, 0]) == vote_index([s["vote_key"] for s in blind]) == 1
    weights = [1, 2, 0, 0, 0]
    assert rank_index(blind, weights) == 0
    for row in rows:
        row["correct"] = not row["correct"]
        row["gold"] = "new reference"
    assert evidence(rows) == blind
    assert rank_index(evidence(rows), weights) == 0
    assert rank_index(evidence([sample(None, 30, False)]), weights) is None


def test_development_prefers_regularization_when_validation_ties():
    rows = [{"grpo": [sample("number:1", 100, True), sample("number:2", 1000, False)]} for _ in range(6)]
    config = {"dev_split_seed": 1, "dev_folds": 3, "penalties": [1.0, 0.1, 0.01]}
    report = develop(rows, config)
    assert report["selected_penalty"] == 1.0
    assert [x["correct"] for x in report["oof"]] == [6, 6, 6]
    assert report["weights"][1] < 0


def test_paired_report_counts_swaps_and_shared_samples():
    samples = [sample("number:1", 10, False, truncated=True), sample("number:2", 20, True)]
    report = summarize([
        dict(samples=samples, vote_index=0, rank_index=1),
        dict(samples=samples, vote_index=1, rank_index=0),
        dict(samples=samples, vote_index=1, rank_index=1),
    ], {"bootstrap_seed": 3, "bootstrap_replicates": 40, "minimum_gain_pp": 2})
    assert report["gains"] == report["losses"] == 1
    assert report["delta_pp"] == 0 and not report["advancement_gate"]
    assert report["changed_picks"] == 2
    assert report["generation_tokens"] == 90 and report["truncated"] == 3


def test_child_seeds_have_disjoint_question_ranges():
    sampling = {"seed": 80, "n": 8}
    seeds = {question_seed(sampling, i) + k for i in range(300) for k in range(8)}
    assert len(seeds) == 2400
    assert min(seeds) == 80 and max(seeds) == 2479
