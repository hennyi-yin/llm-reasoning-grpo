"""Verifiable correctness (1.0) and format (0.1) rewards for TRL."""

from collections.abc import Sequence
from .extract import BOXED, last_boxed, numeric_equal


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, Sequence) or isinstance(completion, (bytes, bytearray)):
        raise TypeError("completion must be text or a list of assistant messages")
    messages = list(completion)
    if not messages or any(not isinstance(m, dict) or not isinstance(m.get("content"), str)
                           for m in messages):
        raise ValueError("completion messages must contain string content")
    return "\n".join(m["content"] for m in messages)


def correctness_reward(completions, gold, **kwargs) -> list[float]:
    if len(completions) != len(gold):
        raise ValueError("completions and gold must have identical lengths")
    return [1.0 if numeric_equal(last_boxed(completion_text(c)), g) else 0.0
            for c, g in zip(completions, gold)]


def format_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        text = completion_text(completion)
        answer = last_boxed(text)
        rewards.append(0.1 if text.count(BOXED) == 1 and answer is not None
                       and answer.strip() else 0.0)
    return rewards
