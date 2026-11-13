from src.math_sft_conservative import summarize


POLICY = dict(preservation_margin_pp=3, min_gain_pp=2, max_unresolved_fraction=0.25,
              max_extra_unresolved_pp=3)


def row(a, b, *, old_b=None, unresolved=False):
    def outcome(ok, previous, via):
        return dict(correct=ok, previous=previous, strict=previous, source=via,
                    truncated=False, n_tokens=100)
    return dict(base=outcome(a, a, "boxed"),
                conservative=outcome(b, b if old_b is None else old_b,
                                     "unresolved" if unresolved else "boxed"))


def test_equivalence_to_base_is_not_a_confirmation_trigger():
    result = summarize("dev", [row(True, True)] * 60 + [row(False, False)] * 40, POLICY)
    assert result["preservation_supported"]
    assert not result["improvement_gate"]
    assert result["decision"] == "stop"


def test_positive_gate_requires_improvement_in_both_sensitivity_judges():
    same = [row(True, True)] * 50 + [row(False, False)] * 40
    assert summarize("dev", same + [row(False, True, old_b=False)] * 10, POLICY)["decision"] == "stop"
    assert summarize("dev", same + [row(False, True)] * 10, POLICY)["decision"] == "confirm"


def test_coverage_can_block_a_promising_point_estimate():
    rows = [row(True, True)] * 50 + [row(False, True)] * 10
    rows += [row(False, False, unresolved=True)] * 10 + [row(False, False)] * 30
    assert summarize("dev", rows, POLICY)["decision"] == "stop"
