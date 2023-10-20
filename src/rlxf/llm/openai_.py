from __future__ import annotations

import os
from typing import Any, List, Literal, Optional

import openai

from rlxf.prompts.response_ranking import Rank
from rlxf.prompts.templates.openai_ import GPT4ResponseRankingPrompt


class OpenAILLM:
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
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.model = model

        openai.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        assert (
            openai.api_key is not None
        ), "Either the `openai_api_key` arg or the `OPENAI_API_KEY` environment variable must be set to use the OpenAI API."

    def generate(self, prompt: str, responses: List[str]) -> Any:
        prompt_template = GPT4ResponseRankingPrompt(
            ranks=[
                Rank(rank=1, description="Correct"),
                Rank(rank=2, description="Incorrect"),
            ],
            ranks_description="The ranking should be based on the correctness of the answer.",
        )
        generated_prompt = prompt_template.generate_prompt(
            instruction=prompt,
            responses=responses,
            for_chat=True,
        )
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=generated_prompt,
        )
        output = response["choices"][0]["message"]["content"].strip()
        return prompt_template.parse_output(output)
