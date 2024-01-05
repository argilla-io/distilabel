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

import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _TOGETHER_AVAILABLE

if _TOGETHER_AVAILABLE:
    import together

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats


logger = get_logger()


class TogetherInferenceLLM(LLM):
    def __init__(
        self,
        task: "Task",
        model: str,
        api_key: Union[str, None] = None,
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 1,
        stop: Union[List[str], None] = None,
        logprobs: int = 0,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the OpenAILLM class.

        Args:
            task (Task): the task to be performed by the LLM.
            model (str): the model to be used for generation.
            max_new_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to 128.
            temperature (float, optional): the temperature to be used for generation. From the Together
                Inference docs: "A decimal number that determines the degree of randomness in the response.
                A value of 0 will always yield the same output. A temperature much less than 1 favors more
                correctness and is appropriate for question answering or summarization. A value approaching
                1 introduces more randomness in the output.". Defaults to 1.0.
            repetition_penalty (float, optional): the repetition penalty to be used for generation. From the
                Together Inference docs: "Controls the diversity of generated text by reducing the likelihood
                of repeated sequences. Higher values decrease repetition.". Defaults to 1.0.
            top_p (float, optional): the top-p value to be used for generation. From the Together
                Inference docs: "used to dynamically adjust the number of choices for each predicted
                token based on the cumulative probabilities. It specifies a probability threshold,
                below which all less likely tokens are filtered out. This technique helps to maintain
                diversity and generate more fluent and natural-sounding text.". Defaults to 1.0.
            top_k (int, optional): the top-k value to be used for generation. From the Together Inference
                docs: "used to limit the number of choices for the next predicted word or token. It specifies
                the maximum number of tokens to consider at each step, based on their probability of occurrence.
                This technique helps to speed up the generation process and can improve the quality of the
                generated text by focusing on the most likely options.". Defaults to 1.
            stop (List[str], optional): strings to delimitate the generation process, so that when the
                model generates any of the provided characters, the generation process is considered completed.
                Defaults to None.
            logprobs (int, optional): the number of logprobs to be returned for each token. From the
                Together Inference docs: "An integer that specifies how many top token log probabilities
                are included in the response for each token generation step.". Defaults to None.
            num_threads (Union[int, None], optional): the number of threads to be used
                for parallel generation. If `None`, no parallel generation will be performed.
                Defaults to `None`.
            prompt_format (Union[SupportedFormats, None], optional): the format to be used
                for the prompt. If `None`, the default format of the task will be used, available
                formats are `openai`, `chatml`, `llama2`, `zephyr`, and `default`. Defaults to `None`,
                but `default` (concatenation of `system_prompt` and `formatted_prompt` with a line-break)
                will be used if no `prompt_formatting_fn` is provided.
            prompt_formatting_fn (Union[Callable[..., str], None], optional): a function to be
                applied to the prompt before generation. If `None`, no formatting will be applied.
                Defaults to `None`.

        Raises:
            AssertionError: if the provided `model` is not available in Together Inference.

        Examples:
            >>> from distilabel.tasks.text_generation import TextGenerationTask as Task
            >>> from distilabel.llm import TogetherInferenceLLM
            >>> task = Task()
            >>> llm = TogetherInferenceLLM(model="togethercomputer/llama-2-7b", task=task, prompt_format="llama2")
        """
        if not _TOGETHER_AVAILABLE:
            raise ImportError(
                "`TogetherInferenceLLM` cannot be used as `together` is not installed, please "
                " install it with `pip install together`."
            )

        together.api_key = api_key or os.getenv("TOGETHER_API_KEY", None)
        if together.api_key is None:
            raise ValueError(
                "No `api_key` provided, please provide one or set the `TOGETHER_API_KEY` "
                "environment variable."
            )

        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        assert (
            model in self.available_models
        ), f"Provided `model` is not available in Together Inference, available models are {self.available_models}"
        self.model = model

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.stop = stop
        self.logprobs = logprobs

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "repetition_penalty": self.repetition_penalty,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stop": self.stop,
                "logprobs": self.logprobs,
            },
        )

    @cached_property
    def available_models(self) -> List[str]:
        """Returns the list of available models in Together Inference."""
        # TODO: exclude the image models
        return [model["name"] for model in together.Models.list()]

    @property
    def model_name(self) -> str:
        """Returns the name of the Together Inference model."""
        return self.model

    def _generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
    ) -> List[List[LLMOutput]]:
        """Generates `num_generations` for each input in `inputs`.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            num_generations (int, optional): the number of generations to be performed for each
                input. Defaults to 1.

        Returns:
            List[List[LLMOutput]]: the generated outputs.
        """
        prompts = self._generate_prompts(inputs, default_format=None)
        outputs = []
        for prompt in prompts:
            batch = []
            for _ in range(num_generations):
                output = together.Complete.create(
                    prompt=prompt,
                    model=self.model,
                    max_tokens=self.max_new_tokens,
                    stop=self.stop,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    logprobs=self.logprobs,
                )
                if output["output"]["choices"] is not None:
                    for choice in output["output"]["choices"]:
                        try:
                            parsed_response = self.task.parse_output(
                                choice["text"].strip()
                            )
                        except Exception as e:
                            logger.error(
                                f"Error parsing Together Inference response: {e}"
                            )
                            parsed_response = None
                        batch.append(
                            LLMOutput(
                                model_name=self.model_name,
                                prompt_used=prompt,
                                raw_output=choice["text"],
                                parsed_output=parsed_response,
                            )
                        )
            if len(batch) > 0:
                outputs.append(batch)
        return outputs
