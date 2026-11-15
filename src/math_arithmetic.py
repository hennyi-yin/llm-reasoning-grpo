"""Check supported numerical equalities in a completion and rerank its answer."""

import ast
from collections import Counter
from fractions import Fraction
from math import isqrt, log2
import operator
import re

from .math_multisample import vote_index
from .math_selector import selected_correct


MATH = re.compile(
    r"\\\[(.*?)\\\]|\\\((.*?)\\\)|(?<!\\)\$\$(.*?)(?<!\\)\$\$|"
    r"(?<![\\$])\$([^\r\n$]+)(?<!\\)\$|"
    r"\\begin\{(align\*?|aligned|gather\*?)\}(.*?)\\end\{\5\}",
    re.S,
)
FRACTION = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")
ROOT = re.compile(r"\\sqrt\{([^{}]+)\}")
ALLOWED = re.compile(r"[0-9r().+*/\- \t\r\n]+")
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def bounded(value):
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 192:
        raise ValueError("Arithmetic result exceeds the verification budget")
    return value


def walk(node, source):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return bounded(Fraction(ast.get_source_segment(source, node)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](walk(node.operand, source))
    if isinstance(node, ast.BinOp):
        a, b = walk(node.left, source), walk(node.right, source)
        if isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 12 and (a or b):
            return bounded(a ** b.numerator)
        if type(node.op) in OPERATORS:
            return bounded(OPERATORS[type(node.op)](a, b))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "r"
            and len(node.args) == 1 and not node.keywords):
        value = walk(node.args[0], source)
        a, b = isqrt(value.numerator), isqrt(value.denominator)
        if a*a == value.numerator and b*b == value.denominator:
            return Fraction(a, b)
    raise ValueError("Unsupported exact arithmetic")


def exact(expression):
    if len(expression) > 180:
        return None
    source = expression.strip()
    for original, replacement in (
        (r"\left", ""), (r"\right", ""), (r"\,", ""), (r"\;", ""), (r"\!", ""),
        (r"\cdot", "*"), (r"\times", "*"), (r"\div", "/"),
        ("×", "*"), ("÷", "/"), ("−", "-"), ("^", "**"),
    ):
        source = source.replace(original, replacement)
    for _ in range(6):
        modified = FRACTION.sub(r"((\1)/(\2))", source)
        modified = ROOT.sub(r"r(\1)", modified)
        if modified == source:
            break
        source = modified
    # Only translated square roots may introduce the internal r() primitive.
    if "r" in expression.replace(r"\sqrt", "").replace(r"\right", "").replace("frac", ""):
        return None
    source = source.replace("{", "(").replace("}", ")").strip()
    if not ALLOWED.fullmatch(source) or "//" in source or re.search(r"\*\s+\*|/\s+/", source):
        return None
    source = re.sub(r"(?<=\d)\s*(?=\(|r\()|(?<=\))\s*(?=[0-9(r])", "*", source)
    try:
        return walk(ast.parse(source, mode="eval").body, source)
    except (SyntaxError, ArithmeticError, ValueError, RecursionError):
        return None


def simple_literal(expression):
    return re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", expression.strip()) is not None


def checks(text, key):
    """Return unique checked arithmetic links and links supporting a numeric key."""
    answer = Fraction(key.removeprefix("number:")) if key and key.startswith("number:") else None
    passed, failed, linked = set(), set(), set()
    for matched in MATH.finditer(text):
        *delimited, _, environment = matched.groups()
        fragment = next((x for x in delimited if x is not None), environment)
        for line in re.split(r"\\\\(?:\[.*?\])?", fragment):
            if re.search(r"\\(?:approx|sim|equiv|bmod|pmod|mod)\b|[≈≡]", line):
                continue
            parts = [x.strip().rstrip(".;,") for x in line.replace("&", "").split("=")]
            values = [exact(part) for part in parts] if len(parts) > 1 else []
            for i, (a, b) in enumerate(zip(values, values[1:])):
                if a is None or b is None:
                    continue
                expr_a, expr_b = parts[i:i+2]
                link = tuple(sorted(re.sub(r"\s+", "", x) for x in (expr_a, expr_b)))
                if a != b:
                    failed.add(link)
                elif not (simple_literal(expr_a) and simple_literal(expr_b)):
                    passed.add(link)
                    if a == answer:
                        linked.add(link)
    return {"checked_true": len(passed), "checked_false": len(failed), "answer_links": len(linked)}


def choose(samples, weight):
    groups = {}
    for i, s in enumerate(samples):
        if s["vote_key"] is not None:
            groups.setdefault(s["vote_key"], []).append(i)
    if not groups:
        return None

    def score(indexes):
        quality = sum(int(samples[i]["answer_links"] > 0) - int(samples[i]["checked_false"] > 0)
                      for i in indexes) / len(indexes)
        return log2(len(indexes)) + weight * quality

    return max(groups.values(), key=score)[0]


def develop(rows, config):
    def count(weight):
        return sum(selected_correct(r["samples"], choose(r["samples"], weight)) for r in rows)

    votes = sum(selected_correct(r["samples"], vote_index([s["vote_key"] for s in r["samples"]])) for r in rows)
    scores = [{"weight": w, "correct": count(w)} for w in config["verification_weights"]]
    winner = max(scores, key=lambda r: r["correct"])
    return {"questions": len(rows), "vote_correct": votes, "scores": scores,
            "selected_weight": winner["weight"]}


def diagnostics(rows):
    counts = Counter(checked_samples=0, contradictory_samples=0, linked_samples=0,
                     checked_questions=0, lower_vote_swaps=0,
                     lower_vote_gains=0, lower_vote_losses=0)
    for row in rows:
        samples = row["samples"]
        checked = [bool(s["checked_true"] or s["checked_false"]) for s in samples]
        counts["checked_samples"] += sum(checked)
        counts["checked_questions"] += any(checked)
        counts["contradictory_samples"] += sum(s["checked_false"] > 0 for s in samples)
        counts["linked_samples"] += sum(s["answer_links"] > 0 for s in samples)
        vote, rank = row["vote_index"], row["rank_index"]
        if vote is None or rank is None or vote == rank:
            continue
        support = Counter(s["vote_key"] for s in samples)
        if support[samples[rank]["vote_key"]] < support[samples[vote]["vote_key"]]:
            counts["lower_vote_swaps"] += 1
            before, after = selected_correct(samples, vote), selected_correct(samples, rank)
            counts["lower_vote_gains"] += after and not before
            counts["lower_vote_losses"] += before and not after
    return dict(counts)
