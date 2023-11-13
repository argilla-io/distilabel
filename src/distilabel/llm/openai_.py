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

from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from openai import OpenAI

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class OpenAILLM(LLM):
    def __init__(
        self,
        task: "Task",
        model: str = "gpt-3.5-turbo",
        client: Union[OpenAI, None] = None,
        openai_api_key: Union[str, None] = None,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        self.max_tokens = max_new_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p

        self.client = client or OpenAI(api_key=openai_api_key, max_retries=6)

        assert (
            model in self.available_models
        ), f"Provided `model` is not available in your OpenAI account, available models are {self.available_models}"
        self.model = model

    @cached_property
    def available_models(self) -> List[str]:
        return [model.id for model in self.client.models.list().data]

    @property
    def model_name(self) -> str:
        return self.model

    def _generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
    ) -> List[List[LLMOutput]]:
        prompts = self._generate_prompts(
            inputs, default_format="openai", expected_output_type=list
        )
        outputs = []
        for prompt in prompts:
            chat_completions = self.client.chat.completions.create(
                messages=prompt,
                model=self.model,
                n=num_generations,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=50,
            )

            output = []
            for chat_completion in chat_completions.choices:
                try:
                    parsed_response = self.task.parse_output(
                        chat_completion.message.content.strip()
                    )
                except Exception as e:
                    logger.error(f"Error parsing OpenAI response: {e}")
                    parsed_response = None
                output.append(
                    LLMOutput(
                        prompt_used=prompt,
                        raw_output=chat_completion.message.content,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs
