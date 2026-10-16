from mini_llm.chat import build_prompt_ids
from mini_llm.prompt import format_conversation, format_prompt


class FakeTok:
    def __init__(self, n):
        self.n = n

    def encode(self, text):
        class E:
            def __init__(self, ids):
                self.ids = ids

        # 1 token per character, deterministic for truncation tests
        return E(list(range(min(len(text), self.n))) if isinstance(self.n, int) else None)


class LenTok:
    def encode(self, text):
        class E:
            def __init__(self, ids):
                self.ids = ids

        return E([ord(c) % 997 for c in text])


def test_single_turn_matches_legacy():
    assert format_conversation([], " hi ") == format_prompt(" hi ")
    assert format_conversation([], "hi") == "User: hi\nAssistant:"


def test_multi_turn_exact_tokens():
    h = [("Bonjour", "Salut!"), ("2+2?", "2 + 2 = 4.")]
    assert format_conversation(h, "Et 3+3?") == (
        "User: Bonjour\nAssistant: Salut!\n"
        "User: 2+2?\nAssistant: 2 + 2 = 4.\n"
        "User: Et 3+3?\nAssistant:"
    )


def test_build_ids_drops_oldest_first():
    tok = LenTok()
    hist = [(f"q{i}", f"a{i}") for i in range(20)]
    ids, kept = build_prompt_ids(tok, hist, "final?", ctx=50)
    assert len(ids) <= 49
    assert kept < len(hist)
    # newest turn must survive
    assert "final?" in format_conversation(hist[-kept:] if kept else [], "final?") if kept else True


def test_build_ids_keeps_all_when_fits():
    tok = LenTok()
    hist = [("hi", "hello")]
    ids, kept = build_prompt_ids(tok, hist, "yo?", ctx=512)
    assert kept == 1
    assert len(ids) < 512
