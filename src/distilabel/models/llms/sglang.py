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

import json
from typing import TYPE_CHECKING, Any, List

from pydantic import PrivateAttr, validate_call

from distilabel.models.llms import LLM
from distilabel.models.llms.typing import GenerateOutput, HiddenState
from distilabel.steps.tasks.typing import ChatType

if TYPE_CHECKING:
    from sglang.srt.server import Runtime


class SGLangLLM(LLM):
    _runtime: "Runtime" = PrivateAttr(None)

    def __init__(
        self,
        model: str,
        log_level: str = "error",
        tensor_parallel_size: int = 1,
        **kwargs,
    ):
        """Initialize SGLang Runtime LLM.

        Args:
            model: Model path or name
            log_level: Logging level (default: "error")
            tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
            **kwargs: Additional arguments passed to SGLang Runtime
        """
        super().__init__()
        from sglang.srt.server import Runtime

        self._runtime = Runtime(
            model_path=model,
            log_level=log_level,
            tp_size=tensor_parallel_size,
            **kwargs,
        )
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @validate_call
    def generate(
        self,
        inputs: List[ChatType],
        num_generations: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> List[GenerateOutput]:
        """Generate completions for the input prompts.

        Args:
            inputs: List of chat messages
            num_generations: Number of generations per prompt
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling threshold
            **kwargs: Additional sampling parameters

        Returns:
            List of generation outputs
        """
        # Convert chat messages to prompt string
        prompts = []
        for messages in inputs:
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            prompts.append(prompt.strip())

        # Set up sampling parameters
        sampling_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            **kwargs,
        }

        # Generate completions
        outputs = []
        for _ in range(num_generations):
            response = self._runtime.generate(
                prompt=prompts, sampling_params=sampling_params
            )
            parsed = json.loads(response)

            for completion in parsed:
                outputs.append(
                    {
                        "text": completion["text"],
                        "tokens": completion.get("token_ids", []),
                        "logprobs": completion.get("logprobs", None),
                    }
                )

        return outputs

    def get_last_hidden_state(self, inputs: List[ChatType]) -> List[HiddenState]:
        """Get hidden states for input prompts.

        Args:
            inputs: List of chat messages

        Returns:
            List of hidden states
        """
        raise NotImplementedError("Hidden state extraction not supported for SGLang")

    def __del__(self):
        """Cleanup runtime when object is deleted."""
        if hasattr(self, "runtime"):
            self.runtime.shutdown()
