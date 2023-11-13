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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger

if TYPE_CHECKING:
    from llama_cpp import Llama

    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class LlamaCppLLM(LLM):
    def __init__(
        self,
        model: "Llama",
        task: "Task",
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        prompt_format: Union[SupportedFormats, None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        super().__init__(
            task=task,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        self.max_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty

        self.model = model

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List[LLMOutput]]:
        prompts = self._generate_prompts(
            inputs, default_format="llama2", expected_output_type=str
        )
        outputs = []
        for prompt in prompts:
            output = []
            for _ in range(num_generations):
                raw_output = self.model.create_completion(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repeat_penalty=self.repeat_penalty,
                )
                try:
                    parsed_output = self.task.parse_output(
                        raw_output["choices"][0]["text"].strip()
                    )
                except Exception as e:
                    logger.error(f"Error parsing llama-cpp output: {e}")
                    parsed_output = None
                output.append(
                    LLMOutput(
                        prompt_used=prompt,
                        raw_output=raw_output,
                        parsed_output=parsed_output,
                    )
                )
            outputs.append(output)
        return outputs
