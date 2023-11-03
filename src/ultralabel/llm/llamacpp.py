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

import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

from ultralabel.llm.base import LLM

if TYPE_CHECKING:
    from llama_cpp import Llama

    from ultralabel.tasks.base import Task


class LlamaCppLLM(LLM):
    def __init__(
        self,
        model: "Llama",
        task: "Task",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        super().__init__(
            task=task,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            formatting_fn=formatting_fn,
        )

        self.model = model

    def _generate(
        self, input: Dict[str, Any], num_generations: int = 1
    ) -> Tuple[Any, List[Any]]:
        prompt = self.task.generate_prompt(**input)
        if self.formatting_fn is not None:
            prompt = self.formatting_fn(prompt)
        raw_outputs, parsed_outputs = [], []
        for _ in range(num_generations):
            raw_output = self.model.create_completion(
                prompt, max_tokens=self.max_new_tokens, temperature=self.temperature
            )
            raw_outputs.append(raw_output)
            try:
                parsed_output = self.task.parse_output(
                    raw_output["choices"][0]["text"].strip()
                )
            except Exception as e:
                warnings.warn(
                    f"Error parsing llama-cpp output: {e}", UserWarning, stacklevel=2
                )
                parsed_output = {}
            parsed_outputs.append(parsed_output)
        return raw_outputs, parsed_outputs
