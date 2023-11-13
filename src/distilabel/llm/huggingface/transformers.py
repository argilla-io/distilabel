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

import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Union

import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from distilabel.llm.base import LLM
from distilabel.logger import get_logger
from distilabel.tasks.prompt import Prompt

if TYPE_CHECKING:
    from distilabel.llm.utils import LLMOutput
    from distilabel.tasks.base import Task

logger = get_logger()


class TransformersLLM(LLM):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: "Task",
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: Union[float, None] = None,
        top_k: Union[int, None] = None,
        top_p: Union[float, None] = None,
        typical_p: Union[float, None] = None,
        num_threads: Union[int, None] = None,
        prompt_format: Union[
            Literal["llama2", "openai", "chatml", "zephyr"], None
        ] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.typical_p = typical_p

        self.model = model
        if self.device != "cpu":
            self.model.to(self.device)

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if (
            hasattr(self.tokenizer, "use_default_system_prompt")
            and self.tokenizer.use_default_system_prompt
        ):
            # The `tokenizer` also has a method named `apply_chat_template` that expects a `Conversation` as OpenAI does with the ChatML format
            warnings.warn(
                "The provided `tokenizer` has `use_default_system_prompt=True` which means that the default system prompt will be used, which may collide with the `task` provided as an arg to this class.",
                UserWarning,
                stacklevel=2,
            )

    @cached_property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    def _generate(
        self, input: Dict[str, Any], num_generations: int = 1
    ) -> List[LLMOutput]:
        prompt = self.task.generate_prompt(**input)
        if not isinstance(prompt, Prompt) and self.prompt_formatting_fn is not None:
            warnings.warn(
                f"The method `generate_prompt` is not returning a `Prompt` class but a prompt of `type={type(prompt)}`, meaning that a pre-formatting has already been applied in the `task.generate_prompt` method, so the usage of a `formatting_fn` is discouraged.",
                UserWarning,
                stacklevel=2,
            )
            prompt = self.prompt_formatting_fn(prompt)
        elif isinstance(prompt, Prompt) and self.prompt_formatting_fn is None:
            if self.prompt_format:
                prompt = prompt.format_as(format=self.prompt_format)  # type: ignore
            else:
                prompt = f"{prompt.system_prompt}\n{prompt.formatted_prompt}"
        if not isinstance(prompt, str):
            raise ValueError(
                f"The provided `prompt={prompt}` is of `type={type(prompt)}`, but it must be a `str`, make sure that `task.generate_prompt` returns a `str` or that the `formatting_fn` formats the prompt as a `str`."
            )
        encoding = self.tokenizer(text=prompt, padding=True, return_tensors="pt")
        if self.device != "cpu":
            encoding = encoding.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **encoding,
                generation_config=GenerationConfig(
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    typical_p=self.typical_p,
                    num_generations=num_generations,
                ),
            )
        raw_outputs = self.tokenizer.batch_decode(
            generated_ids[:, -(encoding.input_ids.shape[1]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        outputs = []
        for raw_output in raw_outputs:
            try:
                parsed_output = self.task.parse_output(raw_output)
            except Exception as e:
                logger.error(f"Error parsing Transformers output: {e}")
                parsed_output = None
            outputs.append(
                LLMOutput(
                    prompt_used=prompt,
                    raw_output=raw_output,
                    parsed_output=parsed_output,
                )
            )
        return outputs
