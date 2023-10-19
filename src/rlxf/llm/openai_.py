from __future__ import annotations

import os
from typing import Any, Callable, Generator, Literal, Optional

import openai

from rlxf.llm.base import LLM
from rlxf.prompts.openai_ import GPT4Prompt


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
        openai_api_key: Optional[str] = None,
        prompt_formatting_fn: Optional[Callable] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> None:
        self.model = model
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        assert (
            self.openai_api_key is not None
        ), "Either the `openai_api_key` arg or the `OPENAI_API_KEY` environment variable must be set to use the OpenAI API."
        openai.api_key = self.openai_api_key

        self.prompt_formatting_fn = prompt_formatting_fn
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_return_sequences = num_return_sequences

    def batch_generate(
        self, prompts: list[str], responses: list[list[str]] | None = None
    ) -> Generator[list[str], None, None]:
        for prompt, responses_ in zip(prompts, responses):
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.prompt_formatting_fn(prompt, responses_),
                temperature=0,
                max_tokens=self.max_new_tokens,
                n=self.num_return_sequences,
            )
            if len(response["choices"]) > 1 and self.num_return_sequences > 1:
                yield [
                    choice["message"]["content"].strip()
                    for choice in response["choices"]
                ]
            else:
                yield response["choices"][0]["message"]["content"].strip()

        # # Extract the rating and rationale from the response
        # rating_output = response["choices"][0]["message"]["content"].strip()

        # sections = rating_output.split("#### Output for Text ")[
        #     1:
        # ]  # Ignore any content before the first header
        # parsed_output = []
        # for section in sections:
        #     _, rating_line, rationale_line = section.strip().split("\n", 2)
        #     rating = rating_line.split(": ")[1]
        #     rationale = rationale_line.split(": ")[1]
        #     parsed_output.append({"rating": rating, "rationale": rationale})

    @classmethod
    def as_generator(cls, model: str, **kwargs: Any) -> "OpenAILLM":
        """Classmethod with some helper defaults to act as a response generator for any
        given prompt.
        """
        raise NotImplementedError(
            "`as_generator` is not implemented yet for `OpenAILLM`"
        )

    @classmethod
    def as_ranker(cls, model: str, **kwargs: Any) -> "OpenAILLM":
        """Classmethod with some helper defaults to act as a response ranker for any
        given collection of responses.
        """
        return cls(model=model, prompt_formatting_fn=GPT4Prompt.rank_format, **kwargs)
