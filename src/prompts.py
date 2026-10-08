"""The single frozen conversation template for training and inference."""

INSTRUCTION = "Solve the problem step by step, then give the final answer as \\boxed{answer}."


def build_messages(question: str) -> list[dict[str, str]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    return [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]
