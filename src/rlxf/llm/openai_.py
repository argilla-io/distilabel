from __future__ import annotations

import os
from typing import Generator, List, Literal, Optional

import openai

from rlxf.prompts.ranking import RankingPromptTemplate, Task, Rank


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

    def generate(self, prompt: str, responses: List[str]) -> str:
        prompt_template = RankingPromptTemplate(
            task=Task.QUESTION_ANSWERING,
            ranks=[
                Rank(rank=1, description="Correct"),
                Rank(rank=2, description="Incorrect"),
            ],
            ranks_description="The ranking should be based on the correctness of the answer.",
            instruction=prompt,
            responses=responses,
        )
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "user", "content": str(prompt_template)},
            ],
        )
        output = response["choices"][0]["message"]["content"].strip()
        return prompt_template.process_output(output)
