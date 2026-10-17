from mini_llm.prompt import format_prompt


def test_format_prompt_trims_question():
    assert format_prompt(" hi ") == "User: hi\nAssistant:"
