from fractions import Fraction

from src.math_arithmetic import checks, choose, diagnostics, exact


def test_exact_arithmetic_is_bounded_and_rejects_unknown_syntax():
    assert exact(r"\frac{3^2+7}{\sqrt{16}}") == 4
    assert exact(r"2\left(1+\frac{1}{4}\right)") == Fraction(5, 2)
    assert exact(r"0.1 + 0.2") == Fraction(3, 10)
    assert exact(r"-2^{2}") == -4
    for expr in ("x+1", "r(4)", r"\sqrt{2}", "2^100", "1/0", "7//2", "__import__('os')"):
        assert exact(expr) is None


def test_extracts_checked_chain_links_and_counts_each_once():
    text = (r"\[x=2+3=5.\] \(2+3=5\) "
            r"$4\cdot 3=13$ \[1/3\approx 0.33\] \[ x+2=x+3 \] "
            r"\begin{align*} 1+1 &= 2 \\ 2^3 &= 8 \end{align*}")
    assert checks(text, "number:5") == {"checked_true": 3, "checked_false": 1, "answer_links": 1}
    assert checks(r"\[x=\pm3\] \[2^{2}\equiv 1\pmod{3}\]", None) == {
        "checked_true": 0, "checked_false": 0, "answer_links": 0}


def test_low_vote_answer_can_win_without_reference_and_swaps_are_counted():
    wrong = {"vote_key": "number:3", "checked_true": 0, "checked_false": 1, "answer_links": 0,
             "correct": False}
    right = {"vote_key": "number:2", "checked_true": 1, "checked_false": 0, "answer_links": 1,
             "correct": True}
    samples = [wrong, wrong.copy(), wrong.copy(), right]
    assert choose(samples, 0) == 0
    assert choose(samples, 1) == 3
    assert choose([{**s, "correct": not s["correct"]} for s in samples], 1) == 3
    report = diagnostics([{"samples": samples, "vote_index": 0, "rank_index": 3}])
    assert (report["lower_vote_swaps"], report["lower_vote_gains"], report["lower_vote_losses"]) == (1, 1, 0)
