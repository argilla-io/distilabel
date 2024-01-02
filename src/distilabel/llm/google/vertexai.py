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

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from distilabel.llm import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _VERTEXAI_AVAILABLE

if _VERTEXAI_AVAILABLE:
    from google.api_core import exceptions
    from vertexai.language_models import CodeGenerationModel, TextGenerationModel
    from vertexai.preview.generative_models import GenerationConfig, GenerativeModel

    _VERTEXAI_API_RETRY_ON_EXCEPTIONS = (
        exceptions.ResourceExhausted,
        exceptions.ServiceUnavailable,
        exceptions.Aborted,
        exceptions.DeadlineExceeded,
        exceptions.GoogleAPIError,
    )

else:
    _VERTEXAI_API_RETRY_ON_EXCEPTIONS = ()

if TYPE_CHECKING:
    from vertexai.language_models._language_models import (
        MultiCandidateTextGenerationResponse,
    )
    from vertexai.preview.generative_models import GenerationResponse

    from distilabel.tasks.base import Task

_VERTEXAI_API_STOP_AFTER_ATTEMPT = 6
_VERTEXAI_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER = 1
_VERTEXAI_API_WAIT_RANDOM_EXPONENTIAL_MAX = 10

logger = get_logger()


def is_gemini_model(model: str) -> bool:
    """Returns `True` if the model is a model from the Vertex AI Gemini API.

    Args:
        model (str): the model name to be checked.

    Returns:
        bool: `True` if the model is a model from the Vertex AI Gemini API.
    """
    return "gemini" in model


def is_codey_model(model: str) -> bool:
    """Returns `True` if the model is a model from the Vertex AI Codey API.

    Args:
        model (str): the model name to be checked.

    Returns:
        bool: `True` if the model is a model from the Vertex AI Codey API.
    """
    return "code" in model


class VertexAILLM(LLM):
    """An `LLM` which allows to use models from the Vertex AI APIs:

    - Gemini API: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
    - Codey API: https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview
    - Text API: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text

    """

    def __init__(
        self,
        task: "Task",
        model: str = "gemini-pro",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: int = 128,
        stop_sequences: Optional[List[str]] = None,
        num_threads: Union[int, None] = None,
    ) -> None:
        """Initializes the `VertexGenerativeModelLLM` class.

        Args:
            task (Task): the task to be performed by the LLM.
            model (str, optional): the model to be used for generation. Defaults to "gemini-pro".
            temperature (float, optional): the temperature to be used for generation.
                Defaults to 1.0.
            top_p (float, optional): the top-p value to be used for generation.
                Defaults to 1.0.
            top_k (int, optional): the top-k value to be used for generation.
                Defaults to 40.
            max_new_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to 128.
            num_threads (Union[int, None], optional): the number of threads to be used
                for parallel generation. If `None`, no parallel generation will be performed.
                Defaults to `None`.
        """
        super().__init__(task=task, num_threads=num_threads)

        if not _VERTEXAI_AVAILABLE:
            raise ImportError(
                "`VertexAILLM` cannot be used as `google-cloud-aiplatform` is not installed,"
                " please install it with `pip install google-cloud-aiplatform`"
            )

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_new_tokens
        self.stop_sequences = stop_sequences

        # TODO: check if there's any endpoint to get available models
        # if model not in {"gemini-pro", "gemini-pro-vision"}:
        #     raise ValueError(
        #         f"Model '{model}' is not available in the Gemini API of Vertex AI."
        #     )
        if is_gemini_model(model):
            self.model = GenerativeModel(model)
        elif is_codey_model(model):
            self.model = CodeGenerationModel.from_pretrained(model)
        else:
            self.model = TextGenerationModel.from_pretrained(model)

    @property
    def model_name(self) -> str:
        """Returns the name of the model used for generation."""
        if isinstance(self.model, GenerativeModel):
            return self.model._model_name

        return self.model._model_id

    def _generate_contents(
        self, inputs: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Generates a list of valid dicts that can be parsed to `vertexai.preview.generative_models.Content`
        objects for each input.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.

        Returns:
            List[List[Dict[str, Any]]]: the list of valid `vertexai.preview.generative_models.Content`
                objects.
        """
        contents = []
        for input in inputs:
            prompt = self.task.generate_prompt(**input)
            contents.append(
                [
                    {"role": "user", "parts": [{"text": prompt.system_prompt}]},
                    {"role": "model", "parts": [{"text": "Understood."}]},
                    {"role": "user", "parts": [{"text": prompt.formatted_prompt}]},
                ]
            )
        return contents

    def _generate_with_generative_model(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        """Generate `num_generations` for each input in `inputs` using a Vertex AI Gemini
        API model."""
        inputs_contents = self._generate_contents(inputs)
        outputs = []
        for contents in inputs_contents:
            output = []
            # TODO: remove this for-loop once `GenerationConfig.candidate_count` valid range is not [1, 2)
            for _ in range(num_generations):
                response = self._call_generative_model_with_backoff(
                    contents=contents,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_output_tokens=self.max_output_tokens,
                    stop_sequences=self.stop_sequences,
                )

                try:
                    parsed_response = self.task.parse_output(response.text)
                except Exception as e:
                    logger.error(
                        f"Error parsing Vertex AI Gemini API model response: {e}"
                    )
                    parsed_response = None

                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=contents,
                        raw_output=response.text,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs

    @retry(
        retry=retry_if_exception_type(_VERTEXAI_API_RETRY_ON_EXCEPTIONS),
        stop=stop_after_attempt(_VERTEXAI_API_STOP_AFTER_ATTEMPT),
        wait=wait_random_exponential(
            multiplier=_VERTEXAI_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER,
            max=_VERTEXAI_API_WAIT_RANDOM_EXPONENTIAL_MAX,
        ),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
        reraise=True,
    )
    def _call_generative_model_with_backoff(
        self, contents: List[Dict[str, Any]], **kwargs: Any
    ) -> "GenerationResponse":
        return self.model.generate_content(  # type: ignore
            contents=contents,
            # TODO: update `candidate_count` to have `num_generations` as value once valid range is not [1, 2)
            generation_config=GenerationConfig(candidate_count=1, **kwargs),
        )

    def _generate_with_text_generation_model(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        """Generate `num_generations` for each input in `inputs` using a Vertex AI Text/Code
        API model."""
        prompts = self._generate_prompts(inputs, default_format="default")
        outputs = []
        for prompt in prompts:
            response = self._call_text_generation_model(
                prompt=prompt,
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                stop_sequences=self.stop_sequences,
                # WARNING: The model can return < `candidate_count` generations depending
                # on the generation parameters and the input.
                candidate_count=num_generations,
            )

            output = []
            for candidate in response.candidates:
                try:
                    parsed_response = self.task.parse_output(candidate.text)
                except Exception as e:
                    logger.error(
                        f"Error parsing Vertex AI Text/Code API model response: {e}"
                    )
                    parsed_response = None

                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=prompt,
                        raw_output=candidate.text,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)

        return outputs

    @retry(
        retry=retry_if_exception_type(_VERTEXAI_API_RETRY_ON_EXCEPTIONS),
        stop=stop_after_attempt(_VERTEXAI_API_STOP_AFTER_ATTEMPT),
        wait=wait_random_exponential(
            multiplier=_VERTEXAI_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER,
            max=_VERTEXAI_API_WAIT_RANDOM_EXPONENTIAL_MAX,
        ),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
        reraise=True,
    )
    def _call_text_generation_model(
        self, **kwargs: Any
    ) -> "MultiCandidateTextGenerationResponse":
        return self.model.predict(**kwargs)  # type: ignore

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        if isinstance(self.model, GenerativeModel):
            return self._generate_with_generative_model(inputs, num_generations)

        return self._generate_with_text_generation_model(inputs, num_generations)
