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
import re
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

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
    import google.auth
    from google.api_core import exceptions
    from google.api_core.client_options import ClientOptions
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud.aiplatform.gapic import (
        EndpointServiceClient,
        PredictionServiceClient,
    )
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
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
    from distilabel.tasks.prompt import SupportedFormats


_VERTEXAI_API_STOP_AFTER_ATTEMPT = 6
_VERTEXAI_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER = 1
_VERTEXAI_API_WAIT_RANDOM_EXPONENTIAL_MAX = 10

_PARSE_VERTEXAI_ENDPOINT_PREDICTION_REGEX = re.compile(r"Output:\s*\n(.*?)(\n|$)")

logger = get_logger()


_vertexai_retry_decorator = retry(
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
    """An `LLM` which allows to use Google's proprietary models from the Vertex AI APIs:

    - Gemini API: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
    - Codey API: https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview
    - Text API: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text

    To use the `VertexAILLM` is necessary to have configured the Google Cloud authentication
    using one of these methods:

    - Setting `GOOGLE_CLOUD_CREDENTIALS` environment variable
    - Using `gcloud auth application-default login` command
    - Using `vertexai.init` function from the `google-cloud-aiplatform` library
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

    def _generate_contents(self, prompts: List[str]) -> List[List[Dict[str, Any]]]:
        """Generates a list of valid dicts that can be parsed to `vertexai.preview.generative_models.Content`
        objects for each input.

        Args:
            prompts (List[str]): the prompts to be used for generation.

        Returns:
            List[List[Dict[str, Any]]]: the list of valid `vertexai.preview.generative_models.Content`
                objects.
        """
        return [[{"role": "user", "parts": [{"text": prompt}]}] for prompt in prompts]

    @_vertexai_retry_decorator
    def _call_generative_model_with_backoff(
        self, contents: List[Dict[str, Any]], **kwargs: Any
    ) -> "GenerationResponse":
        return self.model.generate_content(  # type: ignore
            contents=contents,
            # TODO: update `candidate_count` to have `num_generations` as value once valid range is not [1, 2)
            generation_config=GenerationConfig(candidate_count=1, **kwargs),
        )

    def _generative_model_single_output(
        self, contents: List[Dict[str, Any]]
    ) -> LLMOutput:
        raw_output = None
        try:
            response = self._call_generative_model_with_backoff(
                contents=contents,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_output_tokens,
                stop_sequences=self.stop_sequences,
            )
            raw_output = response.text
            parsed_output = self.task.parse_output(raw_output)
        except ValueError as e:
            logger.error(f"Vertex AI Gemini API model didn't return content: {e}")
            return LLMOutput(
                model_name=self.model_name,
                prompt_used=contents,
                raw_output=None,
                parsed_output=None,
            )
        except Exception as e:
            logger.error(f"Error parsing Vertex AI Gemini API model response: {e}")
            parsed_output = None

        return LLMOutput(
            model_name=self.model_name,
            prompt_used=contents,
            raw_output=raw_output,
            parsed_output=parsed_output,
        )

    def _generate_with_generative_model(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        """Generate `num_generations` for each input in `inputs` using a Vertex AI Gemini
        API model."""
        prompts = self._generate_prompts(inputs, default_format="default")
        inputs_contents = self._generate_contents(prompts)
        outputs = []
        for contents in inputs_contents:
            output = []
            # TODO: remove this for-loop once `GenerationConfig.candidate_count` valid range is not [1, 2)
            for _ in range(num_generations):
                output.append(self._generative_model_single_output(contents=contents))
            outputs.append(output)
        return outputs

    @_vertexai_retry_decorator
    def _call_text_generation_model(
        self, **kwargs: Any
    ) -> "MultiCandidateTextGenerationResponse":
        return self.model.predict(**kwargs)  # type: ignore

    def _text_generation_model_single_output(
        self, prompt: str, num_generations: int
    ) -> List[LLMOutput]:
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
        return output

    def _generate_with_text_generation_model(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        """Generate `num_generations` for each input in `inputs` using a Vertex AI Text/Code
        API model."""
        prompts = self._generate_prompts(inputs, default_format="default")
        outputs = []
        for prompt in prompts:
            outputs.append(
                self._text_generation_model_single_output(prompt, num_generations)
            )
        return outputs

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        if isinstance(self.model, GenerativeModel):
            return self._generate_with_generative_model(inputs, num_generations)

        return self._generate_with_text_generation_model(inputs, num_generations)


class VertexAIEndpointLLM(LLM):
    """An `LLM` which uses a Vertex AI Online prediction endpoint for the generation.

    More information about Vertex AI Endpoints can be found here:
    https://cloud.google.com/vertex-ai/docs/general/deployment#deploy_a_model_to_an_endpoint
    """

    def __init__(
        self,
        endpoint_id: str,
        task: "Task",
        project: Optional[str] = None,
        location: str = "us-central1",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        prompt_argument: str = "prompt",
        num_generations_argument: str = "n",
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the `VertexAIEndpointLLM` class.

        Args:
            endpoint_id (str): the ID of the Vertex AI endpoint to be used for generation.
            task (Task): the task to be performed by the LLM.
            project (Optional[str], optional): the project to be used for generation. If `None`,
                the default project will be used. Defaults to `None`.
            location (str, optional): the location of the Vertex AI endpoint to be used for
                generation. Defaults to "us-central1".
            generation_kwargs (Optional[Dict[str, Any]], optional): the generation parameters
                to be used for generation. The name of the parameters will depend on the
                Docker image used to deploy the model to the Vertex AI endpoint. Defaults
                to `None`.
            prompt_argument (str, optional): the name of the Vertex AI Endpoint key to
                be used for the prompt. Defaults to "prompt".
            num_generations_argument (str, optional): the name of the Vertex AI Endpoint
                key to be used to specify the number of generations per prompt. Defaults
                to "n".
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
        """
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        if not _VERTEXAI_AVAILABLE:
            raise ImportError(
                "`VertexAIEndpointLLM` cannot be used as `google-cloud-aiplatform` is not"
                " installed, please install it with `pip install google-cloud-aiplatform`"
            )

        if project is None:
            try:
                project = google.auth.default()[1]
            except DefaultCredentialsError as e:
                raise ValueError(
                    "No `project` was specified and no default credentials were found."
                ) from e

        if generation_kwargs is None:
            generation_kwargs = {}

        self.endpoint_id = endpoint_id
        self.project = project
        self.location = location
        self.generation_kwargs = generation_kwargs
        self.prompt_argument = prompt_argument
        self.num_generations_argument = num_generations_argument

        self.client = PredictionServiceClient(
            client_options=ClientOptions(
                api_endpoint=f"{self.location}-aiplatform.googleapis.com"
            )
        )

    @cached_property
    def model_name(self) -> str:
        """Returns the name of the model used for generation."""
        client = EndpointServiceClient(
            client_options=ClientOptions(
                api_endpoint=f"{self.location}-aiplatform.googleapis.com"
            )
        )
        endpoint = client.get_endpoint(name=self.endpoint_path)
        return endpoint.deployed_models[0].display_name

    @property
    def endpoint_path(self) -> str:
        """Returns the path of the Vertex AI endpoint to be used for generation."""
        return self.client.endpoint_path(
            project=self.project,  # type: ignore
            location=self.location,
            endpoint=self.endpoint_id,
        )

    @_vertexai_retry_decorator
    def _call_vertexai_endpoint(self, instances: List[Any]) -> Any:
        return self.client.predict(endpoint=self.endpoint_path, instances=instances)

    def _prepare_instances(
        self, prompts: List[str], num_generations: int
    ) -> List["Value"]:
        """Prepares the instances to be sent to the Vertex AI endpoint.

        Args:
            prompts (List[str]): the prompts to be used for generation.
            num_generations (int): the number of generations to be performed for each prompt.

        Returns:
            The instances to be sent to the Vertex AI endpoint.
        """
        instances = []
        for prompt in prompts:
            instance = json_format.ParseDict(
                {
                    self.prompt_argument: prompt,
                    self.num_generations_argument: num_generations,
                    **self.generation_kwargs,
                },
                Value(),
            )
            instances.append(instance)
        return instances

    def _single_output(self, instance: Any) -> List[LLMOutput]:
        try:
            # NOTE: `predict` method accepts a list of instances, but depending on the
            # deployed Docker image, it can just accept one instance.
            response = self._call_vertexai_endpoint(instances=[instance])
        except exceptions.InternalServerError as e:
            raise ValueError(
                "The Vertex AI endpoint returned 500 Internal Server Error. This is"
                " usually caused due to wrong generation parameters. Please check the"
                " `generation_parameters` and try again."
            ) from e

        output = []
        for prediction in response.predictions:
            # Vertex endpoint output is `Prompt:\n{{ model_prompt }}\nOutput:\n{{ model_output }}`
            # so we need to do a pre-parsing to remove the `Prompt:` and `Output:` parts.
            match = _PARSE_VERTEXAI_ENDPOINT_PREDICTION_REGEX.search(prediction)
            if not match:
                raise ValueError(
                    "Couldn't parse the response from the Vertex AI endpoint."
                )

            model_output = match.group(1).strip()

            try:
                parsed_output = self.task.parse_output(model_output)
            except Exception as e:
                logger.error(f"Error parsing Vertex AI endpoint model response: {e}")
                parsed_output = None
            output.append(
                LLMOutput(
                    model_name=self.model_name,
                    prompt_used=instance.struct_value[self.prompt_argument],
                    raw_output=model_output,
                    parsed_output=parsed_output,
                )
            )
        return output

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        prompts = self._generate_prompts(inputs)
        instances = self._prepare_instances(
            prompts=prompts, num_generations=num_generations
        )
        return [self._single_output(instance) for instance in instances]
