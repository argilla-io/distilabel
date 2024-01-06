# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

import pytest
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.llama import Llama2TextGenerationTask
from distilabel.tasks.text_generation.openai import OpenAITextGenerationTask
from distilabel.tasks.text_generation.self_instruct import SelfInstructTask

from tests.helpers import prompt_llama2_format

if TYPE_CHECKING:
    import re

    from distilabel.tasks.text_generation.base import TextGenerationTask


@pytest.mark.parametrize(
    "task, pattern",
    [
        (
            Llama2TextGenerationTask(system_prompt="You are a helpful assistant."),
            prompt_llama2_format,
        ),
        (
            OpenAITextGenerationTask(system_prompt="You are a helpful assistant."),
            None,
        ),
        (
            SelfInstructTask(),
            None,
        ),
    ],
)
def test_tasks(task: "TextGenerationTask", pattern: "re.Pattern"):
    input = "Generate something my boy"
    prompt = task.generate_prompt(input=input)
    if isinstance(task, OpenAITextGenerationTask):
        assert len(prompt) == 2
        for p in prompt:
            assert isinstance(p, dict)
            assert set(p.keys()) == {"role", "content"}

    elif isinstance(task, Llama2TextGenerationTask):
        assert isinstance(prompt, str)
        match = pattern.match(prompt)
        assert match.group("system_prompt") == "You are a helpful assistant."
        assert match.group("formatted_prompt") == "Generate something my boy"

    elif isinstance(task, SelfInstructTask):
        assert isinstance(prompt, Prompt)
        prompt_text = prompt.formatted_prompt
        assert "Develop 5 user queries" in prompt_text
        assert "# AI Application\nAI assistant" in prompt_text
        assert "# Context\nGenerate something my boy\n\n# Output" in prompt_text
