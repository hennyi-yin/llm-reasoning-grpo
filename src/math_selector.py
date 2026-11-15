"""Fit and apply a reference-blind selector over eight sampled completions."""

from math import log2
import random
from statistics import median

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .math_multisample import vote_index
from .stats import mcnemar_exact


FEATURES = ("log2_votes", "log2_scaled_median_tokens", "finished_fraction",
            "numeric_key", "word_key")


def evidence(samples):
    """Project onto fields available at inference, before any reference is read."""
    return [{k: sample[k] for k in ("vote_key", "n_tokens", "truncated")}
            for sample in samples]


def candidates(samples):
    groups = {}
    for i, sample in enumerate(samples):
        key = sample["vote_key"]
        if key is not None:
            groups.setdefault(key, []).append(i)
    result = []
    for key, indexes in groups.items():
        sizes = [samples[i]["n_tokens"] for i in indexes]
        finished = sum(not samples[i]["truncated"] for i in indexes)
        result.append((indexes[0], (
            log2(len(indexes)), log2(1 + median(sizes) / 256), finished / len(indexes),
            float(key.startswith("number:")), float(key.startswith("word:")),
        )))
    return result


def rank_index(samples, weights):
    ranked = candidates(samples)
    if not ranked:
        return None
    return max(ranked, key=lambda c: sum(w * f for w, f in zip(weights, c[1], strict=True)))[0]


def labeled_pairs(samples):
    correct, incorrect = [], []
    for i, features in candidates(evidence(samples)):
        target = correct if samples[i]["correct"] else incorrect
        target.append(np.asarray(features))
    return [yes - no for yes in correct for no in incorrect]


def fit(rows, penalty):
    # Equal total weight per informative question; regularize toward plurality.
    groups = [pairs for row in rows if (pairs := labeled_pairs(row["grpo"]))]
    if not groups:
        raise ValueError("Development set has no informative question pairs")
    x = np.concatenate(groups)
    share = np.concatenate([np.full(len(pairs), 1 / (len(groups) * len(pairs))) for pairs in groups])
    prior = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    def objective(weights):
        margins = x @ weights
        offset = weights - prior
        loss = share @ np.logaddexp(0, -margins) + 0.5 * penalty * (offset @ offset)
        gradient = x.T @ (-share * expit(-margins)) + penalty * offset
        return loss, gradient

    opt = minimize(objective, prior, jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-10, "ftol": 1e-13})
    if not opt.success:
        raise RuntimeError(opt.message)
    return opt.x.tolist()


def selected_correct(samples, index):
    return index is not None and samples[index]["correct"]


def develop(rows, config):
    order = list(range(len(rows)))
    random.Random(config["dev_split_seed"]).shuffle(order)
    folds = config["dev_folds"]
    oof = {penalty: [None] * len(rows) for penalty in config["penalties"]}
    vote = [vote_index([s["vote_key"] for s in evidence(r["grpo"])]) for r in rows]
    for fold in range(folds):
        validation = order[fold::folds]
        held = set(validation)
        training = [r for i, r in enumerate(rows) if i not in held]
        for penalty, picks in oof.items():
            weights = fit(training, penalty)
            for i in validation:
                picks[i] = rank_index(evidence(rows[i]["grpo"]), weights)

    def count(picks):
        return sum(selected_correct(r["grpo"], i) for r, i in zip(rows, picks, strict=True))

    scores = {penalty: count(picks) for penalty, picks in oof.items()}
    winner = max(config["penalties"], key=scores.__getitem__)
    return {
        "questions": len(rows), "vote_correct": count(vote),
        "oof": [{"penalty": p, "correct": scores[p]} for p in config["penalties"]],
        "selected_penalty": winner,
        "informative_questions": sum(bool(labeled_pairs(r["grpo"])) for r in rows),
        "features": list(FEATURES),
        "weights": fit(rows, winner),
    }


def summarize(rows, config):
    vote = np.array([selected_correct(row["samples"], row["vote_index"]) for row in rows])
    ranked = np.array([selected_correct(row["samples"], row["rank_index"]) for row in rows])
    n = len(rows)
    delta = ranked.astype(int) - vote.astype(int)
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = rng.integers(n, size=(config["bootstrap_replicates"], n))
    low, high = np.percentile(delta[draws].mean(axis=1) * 100, [2.5, 97.5]).tolist()
    gain = float(delta.mean() * 100)
    samples = [s for row in rows for s in row["samples"]]
    return {
        "questions": n, "samples": len(samples),
        "vote_correct": int(vote.sum()), "rank_correct": int(ranked.sum()),
        "coverage": sum(any(s["correct"] for s in row["samples"]) for row in rows),
        "gains": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
        "changed_picks": sum(r["vote_index"] != r["rank_index"] for r in rows),
        "delta_pp": gain, "paired_95_ci_pp": [low, high],
        "mcnemar_p": mcnemar_exact(vote, ranked),
        "advancement_gate": bool(gain >= config["minimum_gain_pp"] and low > 0),
        "generation_tokens": sum(s["n_tokens"] for s in samples),
        "truncated": sum(s["truncated"] for s in samples),
    }
