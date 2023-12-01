import re
from typing import Optional, get_args

import pytest
from distilabel.tasks.prompt import Prompt, SupportedFormats

from tests.helpers import (
    prompt_chatml_format,
    prompt_default_format,
    prompt_llama2_format,
    prompt_zephyr_format,
)


@pytest.mark.parametrize(
    "format, pattern",
    [
        ("default", prompt_default_format),
        ("openai", None),
        ("llama2", prompt_llama2_format),
        ("chatml", prompt_chatml_format),
        ("zephyr", prompt_zephyr_format),
        ("unkn", None),
    ],
)
def test_prompt_formats(format: str, pattern: Optional[re.Pattern]):
    prompt = Prompt(
        system_prompt="You are a helpful assistant.",
        formatted_prompt="What are the first 5 Fibonacci numbers?",
    )
    if format not in get_args(SupportedFormats):
        with pytest.raises(ValueError):
            prompt.format_as(format)
        return
    prompt_text = prompt.format_as(format)
    if format == "openai":
        assert isinstance(prompt_text, list)
        assert len(prompt_text) == 2
        for p in prompt_text:
            assert isinstance(p, dict)
            assert set(p.keys()) == {"role", "content"}

    else:
        match = pattern.match(prompt_text)
        assert match.group("system_prompt") == "You are a helpful assistant."
        assert (
            match.group("formatted_prompt") == "What are the first 5 Fibonacci numbers?"
        )
