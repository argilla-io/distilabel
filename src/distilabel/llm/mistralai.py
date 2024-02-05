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
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Union

from distilabel.llm import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _MISTRALAI_AVAILABLE

if _MISTRALAI_AVAILABLE:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage


if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class MistralAILLM(LLM):
    def __init__(
        self,
        task: "Task",
        model: str = "mistral-medium",
        client: Optional["MistralClient"] = None,
        api_key: Optional[str] = os.environ.get("MISTRALAI_API_KEY"),
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        safe_prompt: bool = False,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the MistralAILLM class.

        Args:
            task (Task): the task to be performed by the LLM.
            model (str, optional): the model to be used for generation. Defaults to "mistral-medium".
            client (MistralClient, optional):
                A MistralClient client to be used for generation. Defaults to None.
            api_key (Optional[str], optional):
                The MistralAI API key to be used for generation. Will try to grab it from the environment variable
                if not informed. Visit "https://docs.mistral.ai/#api-access" for more information.
            max_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to 128.
            temperature (float, optional): the temperature to be used for generation.
                Defaults to 1.0.
            top_p (float, optional): the top-p value to be used for generation. Defaults to 1.0.
            seed (Optional[int], optional): the random seed to use for sampling, e.g. 42. Defaults to None.
            safe_prompt (_type_, optional):
                whether to use safe prompt, e.g. True. Defaults to False.
                Visit "https://docs.mistral.ai/platform/guardrailing/" for more information.
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
            AssertionError: if the provided `model` is not available in your MistralAI account.

        Examples:
            >>> import os
            >>> from distilabel.tasks import TextGenerationTask
            >>> from distilabel.llm import MistralAILLM
            >>> llm = MistralAILLM(model="mistral-medium", task=TextGenerationTask(), api_key=os.getenv("MISTRALAI_API_KEY"))
            >>> llm.generate([{"input": "What's the capital of Spain?"}])
        """
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )
        if not _MISTRALAI_AVAILABLE:
            raise ImportError(
                "`MistralAILLM` cannot be used as `mistralai` is not installed, please "
                " install it with `pip install mistralai`."
            )

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.safe_prompt = safe_prompt

        # Explicitly write the default parameters of the model
        self.client = client or MistralClient(
            api_key=api_key, max_retries=5, timeout=120
        )
        assert (
            model in self.available_models
        ), f"Provided `model` is not available in MistralAI, available models are {self.available_models}"
        self.model = model

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
                "safe_prompt": self.safe_prompt,
            },
        )

    @cached_property
    def available_models(self) -> List[str]:
        """Returns the list of available models in MistralAI."""
        return [model.id for model in self.client.list_models().data]

    @property
    def model_name(self) -> str:
        """Returns the name of the MistralAI model."""
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
        prompts = self._generate_prompts(inputs, default_format="openai")
        # The mistralai format is the same as openai, but needs to be converted to mistralai's ChatMessage (pydantic model)
        prompts = [[ChatMessage(**p) for p in prompt] for prompt in prompts]
        outputs = []
        for prompt in prompts:
            responses = []
            for _ in range(num_generations):
                chat_completion_response = self.client.chat(
                    self.model,
                    messages=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    random_seed=self.seed,
                    safe_prompt=self.safe_prompt,
                )
                responses.append(chat_completion_response)

            output = []
            for response in responses:
                chat_completion = response.choices[0]
                try:
                    parsed_response = self.task.parse_output(
                        chat_completion.message.content.strip()
                    )
                except Exception as e:
                    logger.error(f"Error parsing MistralAI response: {e}")
                    parsed_response = None
                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=[p.model_dump() for p in prompt],
                        raw_output=chat_completion.message.content,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs
