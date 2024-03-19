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
from typing import TYPE_CHECKING, Optional, Union

from pydantic import SecretStr, ValidationError, model_validator

from distilabel.llm.openai import OpenAILLM

if TYPE_CHECKING:
    from distilabel.llm.typing import GenerateOutput
    from distilabel.steps.task.typing import ChatType


class AsyncInferenceEndpointsLLM(OpenAILLM):
    """InferenceEndpoints LLM implementation running the async API client via `openai`.

    Attributes:
        base_url: the base URL to use for the Inference Endpoints API requests.
        model: set as default to "tgi" as the LLM inference will expect an endpoint running
            using TGI as the backend / framework.
        api_key: the API key to authenticate the requests to the Inference Endpoints API, which
            is the same as the Hugging Face Hub token.

    Examples:
        >>> from distilabel.llm.huggingface import AsyncInferenceEndpointsLLM
        >>> llm = AsyncInferenceEndpointsLLM(model_id="model-id")
        >>> llm.load()
        >>> output = await llm.agenerate([{"role": "user", "content": "Hello world!"}])
    """

    model_id: Optional[str] = None

    endpoint_name: Optional[str] = None
    endpoint_namespace: Optional[str] = None

    base_url: Optional[str] = None

    model_display_name: Optional[str] = None

    model: str = "tgi"
    api_key: Optional[SecretStr] = os.getenv("HF_TOKEN", None)  # type: ignore

    _env_var: Optional[str] = "HF_TOKEN"
    _model_name: Optional[str] = None

    @model_validator(mode="after")
    def only_one_of_model_id_endpoint_name_or_base_url_provided(
        self,
    ) -> "AsyncInferenceEndpointsLLM":
        """Validates that only one of `model_id`, `endpoint_name`, or `base_url` is provided."""

        if self.model_id and (not self.endpoint_name and not self.base_url):
            return self
        if self.endpoint_name and (not self.model_id and not self.base_url):
            return self
        if self.base_url and (not self.model_id and not self.endpoint_name):
            return self

        raise ValidationError(
            f"Only one of `model_id`, `endpoint_name`, or `base_url` must be provided. Found"
            f" `model_id`={self.model_id}, `endpoint_name`={self.endpoint_name}, and"
            f" `base_url`={self.base_url}."
        )

    def load(self, api_key: Optional[str] = None) -> None:
        """Loads the `AsyncOpenAI` client to benefit from async requests, running the
        Hugging Face Inference Endpoint underneath via the `/v1/chat/completions` endpoint,
        exposed for the models running on TGI using the `text-generation` task.

        Args:
            api_key: the API key to authenticate the requests to the Inference Endpoints API,
                which is the same as the Hugging Face Hub token.

        Raises:
            ImportError: if the `openai` Python client is not installed.
            ImportError: if the `huggingface-hub` Python client is not installed.
            ValueError: if the model is not currently deployed or is not running the TGI framework.
        """

        try:
            from openai import AsyncOpenAI
        except ImportError as ie:
            raise ImportError(
                "OpenAI Python client is not installed. Please install it using"
                " `pip install openai`."
            ) from ie

        try:
            from huggingface_hub import InferenceClient, get_inference_endpoint
        except ImportError as ie:
            raise ImportError(
                "Hugging Face Hub Python client is not installed. Please install it using"
                " `pip install huggingface-hub`."
            ) from ie

        self.api_key = self._handle_api_key_value(
            self_value=self.api_key,
            load_value=api_key,
            env_var=self._env_var,  # type: ignore
        )

        if self.model_id is not None:
            client = InferenceClient()
            status = client.get_model_status(self.model_id)

            if (
                status.state not in {"Loadable", "Loaded"}
                and status.framework != "text-generation-inference"
            ):
                raise ValueError(
                    f"Model {self.model_id} is not currently deployed or is not running the TGI framework"
                )

            self.base_url = client._resolve_url(
                model=self.model_id, task="text-generation"
            )

        if self.endpoint_name is not None:
            client = get_inference_endpoint(
                name=self.endpoint_name,
                namespace=self.endpoint_namespace,
                token=self.api_key.get_secret_value(),
            )
            if client.status in ["paused", "scaledToZero"]:
                client.resume().wait(timeout=30)

            self.base_url = client.url
            self._model_name = client.repository

        self._aclient = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key.get_secret_value(),
            max_retries=6,
        )

    @property
    def model_name(self) -> Union[str, None]:
        """Returns the model name used for the LLM."""
        return self.model_display_name or self._model_name

    async def agenerate(  # type: ignore
        self,
        input: "ChatType",
        num_generations: int = 1,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 0.8,
    ) -> "GenerateOutput":
        """Generates completions for the given input using the OpenAI async client.

        Args:
            input: a single input in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`, and only works with `1`.
            max_new_tokens: the maximun number of new tokens that the model will generate.
                Defaults to `128`.
            frequence_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            top_k: the top-k value to use for the generation. Defaults to `0.8`, since neither
                `0.0` nor `1.0` are valid values in TGI.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        if num_generations != 1:
            raise ValueError(
                "`AsyncInferenceEndpointsLLM` only supports one generation per input"
            )

        return await super().agenerate(
            input=input,
            num_generations=1,
            max_new_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
        )
