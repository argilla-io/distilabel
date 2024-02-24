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
import warnings
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Union,
)

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _GROQ_AVAILABLE

if _GROQ_AVAILABLE:
    from groq import Groq

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats  # TODO: Do we need to add a supported format for Groq?
                                                          #  Groq uses Llama2-70B and Mixtral 8x7B.

logger = get_logger()


class GroqLLM(LLM):
    def __init__(
        self,
        task: "Task",
        model: str = "llama2-70b-4096",
        client: Union["Groq", None] = None,  # TODO: Verify that this Union works.
                                             #  Haven't checked if Union["Groq", None] is perfectly analogous to Union["OpenAI", None]
        api_key: Union[str, None] = None,
        max_new_tokens: int = 128,  # Have left this as default
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the GroqLLM class.

        Args:
            task (Task): the task to be performed by the LLM.
            model (str, optional): the model to be used for generation. Defaults to "llama2-70b-4096" (4096-token context window).
            client (Union["Groq, None], optional): a Groq client to be used for generation.
                If `None`, a new client will be created. Defaults to `None`.
            api_key (Union[str, None], optional): the Groq API key to be used for generation.
                If `None`, the `GROQ_API_KEY` environment variable will be used. Defaults to `None`.
            max_new_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to 128.
            frequency_penalty (float, optional): the frequency penalty to be used for generation.
                Defaults to 0.0.
            presence_penalty (float, optional): the presence penalty to be used for generation.
                Defaults to 0.0.
            temperature (float, optional): the temperature to be used for generation.
                Defaults to 1.0.
            top_p (float, optional): the top-p value to be used for generation.
                Defaults to 1.0.
            num_threads (Union[int, None], optional): the number of threads to be used
                for parallel generation. If `None`, no parallel generation will be performed.
                Defaults to `None`.
            prompt_format (Union[SupportedFormats, None], optional): the format to be used
                for the prompt. If `None`, the default format of the task will be used, available
                formats are `llama2`, `openai`, `chatml` `zephyr`, and `default`. Defaults to `None`,
                but `default` (concatenation of `system_prompt` and `formatted_prompt` with a line-break)
                will be used if no `prompt_formatting_fn` is provided.
            prompt_formatting_fn (Union[Callable[..., str], None], optional): a function to be
                applied to the prompt before generation. If `None`, no formatting will be applied.
                Defaults to `None`.

        Raises:
            AssertionError: if the provided `model` is not available in your GroqAPI account.

        Examples:
            >>> from distilabel.tasks import TextGenerationTask
            >>> from distilabel.llm import GroqLLM
            >>> llm = GroqLLM(model="llama2-70b-4096", task=TextGenerationTask())
            >>> llm.generate([{"input": "How far is Germany from Madrid?"}])
        """
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        if not _GROQ_AVAILABLE:
            raise ImportError(
                "`GroqLLM` cannot be used as `groq` is not installed, please "
                " install it with `pip install groq`."
            )

        self.max_tokens = max_new_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p

        self.client = client or Groq(api_key=api_key, max_retries=6)  # TODO: Double-check if Groq offers a max_retries parameter.
                                                                       # According to this documentation: https://console.groq.com/docs/openai, they likely do.
        assert (
            model in self.available_models
        ), f"Provided `model` is not available in your Groq API account, available models are {self.available_models}"
        self.model = model

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            {
                "max_tokens": self.max_tokens,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        )

    @cached_property
    def available_models(self) -> List[str]:
        """Returns the list of available models in your Groq API account."""
        # TODO: Double-check if the model.id functionality is offered by Groq just as OpenAI offers it.
        #  As of now, this code should work because Groq ONLY currently offers Llama2-70B and Mixtral-8x7B
        #  with the given context windows.
        try:
            return [model.id for model in self.client.models.list().data]
        except:
            return ["llama2-70b-4096", "mixtral-8x7b-32768"]

    @property
    def model_name(self) -> str:
        """Returns the name of the model being used through Groq API."""
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
        prompts = self._generate_prompts(inputs, default_format="llama2")  # Use llama2 rather than openai as default format, because that's what Groq offers
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
            )   # TODO: Note, why don't we have to add the "model" parameter here? Double-check. It should work as it is, however.

            output = []
            for chat_completion in chat_completions.choices:
                try:
                    parsed_response = self.task.parse_output(
                        chat_completion.message.content.strip()
                    )
                except Exception as e:
                    logger.error(f"Error parsing GroqAPI response: {e}")
                    parsed_response = None
                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=prompt,
                        raw_output=chat_completion.message.content,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs


class JSONOpenAILLM(OpenAILLM):
    def __init__(
        self,
        task: "Task",
        model: str = "gpt-3.5-turbo-1106",
        client: Union["OpenAI", None] = None,
        api_key: Union[str, None] = None,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the JSONOpenAILLM class for generating JSON.

        Args:
            task (Task): the task to be performed by the LLM.
            model (str, optional): the model to be used for generation. Defaults to "gpt-3.5-turbo".
            client (Union[OpenAI, None], optional): an OpenAI client to be used for generation.
                If `None`, a new client will be created. Defaults to `None`.
            api_key (Union[str, None], optional): the OpenAI API key to be used for generation.
                If `None`, the `OPENAI_API_KEY` environment variable will be used. Defaults to `None`.
            max_new_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to 128.
            frequency_penalty (float, optional): the frequency penalty to be used for generation.
                Defaults to 0.0.
            presence_penalty (float, optional): the presence penalty to be used for generation.
                Defaults to 0.0.
            temperature (float, optional): the temperature to be used for generation.
                Defaults to 1.0.
            top_p (float, optional): the top-p value to be used for generation.
                Defaults to 1.0.
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
            AssertionError: if the provided `model` does not support JSON input.

        Examples:
            >>> from distilabel.tasks import TextGenerationTask
            >>> from distilabel.llm import JSONOpenAILLM
            >>> llm = JSONOpenAILLM(task=TextGenerationTask())
            >>> llm.generate([{"input": "json for 'What's the capital of Spain?'"}])
        """
        super().__init__(
            task,
            model=model,
            client=client,
            api_key=api_key,
            max_new_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

    @cached_property
    def available_models(self) -> List[str]:
        """Returns the list of available models in your OpenAI account."""
        all_available_models = [model.id for model in self.client.models.list().data]
        json_supporting_models = [
            "gpt-4-0125-preview",
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-3.5-turbo-1106",
        ]
        available_json_supporting_models = list(
            set(all_available_models) & set(json_supporting_models)
        )
        return available_json_supporting_models

    def _generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
    ) -> List[List[LLMOutput]]:
        """Generates `num_generations` for each input in `inputs` in JSON format.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            num_generations (int, optional): the number of generations to be performed for each
                input. Defaults to 1.

        Returns:
            List[List[LLMOutput]]: the generated outputs.
        """
        prompts = self._generate_prompts(inputs, default_format="openai")
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
                response_format={"type": "json_object"},
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
                try:
                    json.loads(chat_completion.message.content)
                except json.JSONDecodeError:
                    warnings.warn(
                        "The response is not a valid JSON.",
                        UserWarning,
                        stacklevel=2,
                    )
                    parsed_response = None
                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=prompt,
                        raw_output=chat_completion.message.content,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs
