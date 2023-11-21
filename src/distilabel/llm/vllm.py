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

from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _VLLM_AVAILABLE

if _VLLM_AVAILABLE:
    from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm import LLM as _vLLM

    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class vLLM(LLM):
    def __init__(
        self,
        vllm: "_vLLM",
        task: "Task",
        max_new_tokens: int = 128,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the vLLM class.

        Args:
            vllm (_vLLM): the vLLM model to be used.
            task (Task): the task to be performed by the LLM.
            max_new_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to 128.
            presence_penalty (float, optional): the presence penalty to be used for generation.
                Defaults to 0.0.
            frequency_penalty (float, optional): the frequency penalty to be used for generation.
                Defaults to 0.0.
            temperature (float, optional): the temperature to be used for generation.
                Defaults to 1.0.
            top_p (float, optional): the top-p value to be used for generation.
                Defaults to 1.0.
            top_k (int, optional): the top-k value to be used for generation.
                Defaults to -1.
            prompt_format (Union[SupportedFormats, None], optional): the format to be used
                for the prompt. If `None`, the default format of the task will be used, available
                formats are `openai`, `chatml`, `llama2`, `zephyr`, and `default`. Defaults to `None`,
                but `default` (concatenation of `system_prompt` and `formatted_prompt` with a line-break)
                will be used if no `prompt_formatting_fn` is provided.
            prompt_formatting_fn (Union[Callable[..., str], None], optional): a function to be
                applied to the prompt before generation. If `None`, no formatting will be applied.

        Examples:
            >>> from vllm import LLM
            >>> from distilabel.tasks.text_generation import TextGenerationTask as Task
            >>> from distilabel.llm import vLLM
            >>> model = LLM(model="gpt2")
            >>> task = Task()
            >>> llm = vLLM(model=model, task=task)
        """
        super().__init__(
            task=task,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        if not _VLLM_AVAILABLE:
            raise ImportError(
                "`vLLM` cannot be used as `vllm` is not installed, please "
                " install it with `pip install vllm`."
            )

        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_new_tokens

        self.vllm = vllm

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            {
                "max_tokens": self.max_tokens,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
            },
        )

    @property
    def model_name(self) -> str:
        """Returns the name of the vLLM model."""
        return self.vllm.llm_engine.model_config.model  # type: ignore

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List[LLMOutput]]:
        """Generates `num_generations` for each input in `inputs`.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            num_generations (int, optional): the number of generations to be performed for each
                input. Defaults to 1.

        Returns:
            List[List[LLMOutput]]: the outputs of the LLM.
        """
        prompts = self._generate_prompts(
            inputs, default_format=None, expected_output_type=str
        )
        requests = self.vllm.generate(
            prompts,
            SamplingParams(  # type: ignore
                n=num_generations,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
            ),
            use_tqdm=False,  # type: ignore
        )
        outputs = []
        for request, prompt in zip(requests, prompts):
            output = []
            for request_output in request.outputs:
                try:
                    parsed_output = self.task.parse_output(request_output.text)
                except Exception as e:
                    logger.error(f"Error parsing vLLM output: {e}")
                    parsed_output = None
                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=prompt,
                        raw_output=request_output.text,
                        parsed_output=parsed_output,
                    )
                )
            outputs.append(output)
        return outputs
