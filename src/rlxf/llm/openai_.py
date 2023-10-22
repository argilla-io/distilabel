from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

import openai

from rlxf.llm.base import LLM

if TYPE_CHECKING:
    from rlxf.prompts.base import PromptTemplate

class OpenAILLM(LLM):
    def __init__(
        self,
        model: Literal[
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
        ],
        prompt_template: "PromptTemplate",
        openai_api_key: str | None = None,
    ) -> None:
        super().__init__(prompt_template)
        
        self.model = model
        openai.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        assert (
            openai.api_key is not None
        ), "Either the `openai_api_key` arg or the `OPENAI_API_KEY` environment variable must be set to use the OpenAI API."

    def generate(self, prompts: list[dict[str, Any]]) -> Any:
        generations = []
        for prompt in prompts:
            prompt = self.prompt_template.generate_prompt(**prompt)
            output = openai.ChatCompletion.create(
                model=self.model,
                messages=prompt,
            )["choices"][0]["message"]["content"].strip()
            output = self.prompt_template.parse_output(output)
            generations.append(output)
        return generations
