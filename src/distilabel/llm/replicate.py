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
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _REPLICATE_AVAILABLE

if _REPLICATE_AVAILABLE:
    import replicate

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class ReplicateLLM(LLM):
    def __init__(
        self,
        task: "Task",
        endpoint_name: str,
        endpoint_revision: Union[str, None] = None,
        replicate_api_token: Union[str, None] = None,
        generation_kwargs: Union[Dict[str, Any], None] = None,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the OpenAILLM class.

        Args:
            task (Task): the task to be performed by the LLM.
            endpoint_name (str): the name of the Replicate endpoint to be used for generation.
            endpoint_revision (Union[str, None], optional): the revision of the Replicate endpoint
                to be used for generation. If `None`, the main revision will be used. Defaults to `None`.
            replicate_api_token (Union[str, None], optional): the Replicate API token to be used for generation.
                If `None`, the `REPLICATE_API_KEY` environment variable will be used. Defaults to `None`.
            generation_kwargs (Dict[str, Any], optional): the keyword arguments to be used for inference.
                Defined within the specification of the Replicate endpoint. Defaults to `{}`.
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
            AssertionError: if the provided `model` is not available in your OpenAI account.

        Examples:
            >>> from distilabel.tasks.text_generation import TextGenerationTask as Task
            >>> from distilabel.llm import ReplicateLLM
            >>> task = Task()
            >>> llm = ReplicateLLM(endpoint_name="<ORG>/<REPO>", endpoint_revision="latest", task=task)
        """
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        if not _REPLICATE_AVAILABLE:
            raise ImportError(
                "`ReplicateLLM` cannot be used as `replicate` is not installed, please "
                " install it with `pip install replicate`."
            )

        if replicate_api_token is not None:
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_token

        if os.getenv("REPLICATE_API_TOKEN") is None:
            raise ValueError(
                "`REPLICATE_API_TOKEN` environment variable must be set in order to use the"
                " `ReplicateLLM` class."
            )

        self.endpoint_name = endpoint_name
        self.endpoint_revision = endpoint_revision or "main"
        self.generation_kwargs = generation_kwargs

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            self.generation_kwargs,
        )

    @property
    def model_name(self) -> str:
        """Returns the name of the Replicate Endpoint."""
        return f"{self.endpoint_name}:{self.endpoint_revision}"

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
        prompts = self._generate_prompts(inputs)
        outputs = []
        for prompt in prompts:
            parsed_outputs = []
            for _ in range(num_generations):
                output = list(
                    replicate.run(
                        f"{self.endpoint_name}:{self.endpoint_revision}",
                        input={
                            "prompt": prompt,
                            **(self.generation_kwargs or {}),
                        },
                        stream=False,
                    )
                )
                try:
                    parsed_response = self.task.parse_output("".join(output))
                except Exception as e:
                    logger.error(f"Error parsing Replicate Endpoint's response: {e}")
                    parsed_response = None
                parsed_outputs.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=prompt,
                        raw_output=output,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(parsed_outputs)
        return outputs
