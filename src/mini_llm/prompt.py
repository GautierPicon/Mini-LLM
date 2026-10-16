def format_prompt(question):
    return f"User: {question.strip()}\nAssistant:"


def format_conversation(history, question):
    """Format a multi-turn conversation with the exact training tokens.

    history: list of (question, answer) tuples, oldest first.
    Returns: "User: q1\\nAssistant: a1\\n...User: q\\nAssistant:"
    Empty history is identical to format_prompt(question).
    """
    parts = []
    for q, a in history:
        parts.append(f"User: {q.strip()}\nAssistant: {a.strip()}\n")
    parts.append(f"User: {question.strip()}\nAssistant:")
    return "".join(parts)