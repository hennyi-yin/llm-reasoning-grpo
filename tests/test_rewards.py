import pytest
from src.rewards import correctness_reward, format_reward


def completion(text):
    return [{"role": "assistant", "content": text}]


def test_correctness_uses_only_last_answer():
    outputs = [completion(r"\boxed{42}"), completion(r"\boxed{42} \boxed{0}"),
               completion("42"), completion(r"\boxed{42/2}"), completion(r"\boxed{42} \boxed{")]
    assert correctness_reward(outputs, [42] * 5) == [1, 0, 0, 0, 0]


def test_currency_and_decimal_reward():
    assert correctness_reward([completion(r"\boxed{\$1,234.50}")], [1234.5]) == [1]


def test_format_requires_one_complete_nonempty_box():
    outputs = [completion(r"\boxed{1}"), completion(r"\boxed{1} \boxed{2}"),
               completion(r"\boxed{"), completion(r"\boxed{}"), completion("12")]
    assert format_reward(outputs) == [.1, 0, 0, 0, 0]


def test_plain_text_completions_supported():
    assert correctness_reward([r"\boxed{7}"], [7]) == [1]


def test_reward_does_not_silently_truncate_zip():
    with pytest.raises(ValueError):
        correctness_reward([completion(r"\boxed{7}")], [])


def test_invalid_completion_shape():
    with pytest.raises(ValueError):
        format_reward([[{"role": "assistant", "content": None}]])
