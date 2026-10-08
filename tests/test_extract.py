import pytest
from src.extract import last_boxed, to_number, gsm8k_gold, numeric_equal


@pytest.mark.parametrize("text, expected", [
    (r"Work \boxed{1} revised \boxed{2}", "2"),
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"\boxed{\{1,2\}}", r"\{1,2\}"),
    (r"\boxed{1} then \boxed{", None),
    ("The answer is 72", None),
    (r"\boxed{}", ""),
    (r"\boxed{{a}_{b}}", "{a}_{b}"),
])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("1,234", 1234), ("$1,234.50", 1234.5), (r"\$12", 12),
    ("$-12.5$", -12.5), ("-$12.5", -12.5), ("−12", -12),
    (".5", .5), ("  +12.0  ", 12), ("1e3", 1000),
])
def test_complete_numeric_normalization(text, expected):
    assert to_number(text) == expected


@pytest.mark.parametrize("text", [None, "12/2", "12 apples", "answer 12", "12 and 13",
                                  "1,2", "12,34", "1+2", r"\frac{1}{2}",
                                  "NaN", "inf", "", True, "1e9999"])
def test_no_numeric_prefix_acceptance(text):
    assert to_number(text) is None


def test_strict_tolerance():
    assert numeric_equal("0.0000009", "0")
    assert not numeric_equal("0.000001", "0")
    assert not numeric_equal("12/2", "12")


def test_gsm_gold():
    assert gsm8k_gold("Steps with 3 in them\n#### $1,234.50") == 1234.5
    with pytest.raises(ValueError):
        gsm8k_gold("Steps without an answer delimiter")
    with pytest.raises(ValueError):
        gsm8k_gold("Steps\n#### 12/2")
