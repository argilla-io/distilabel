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

import asyncio
import os
from typing import TYPE_CHECKING, Any, List, Optional, Union

from pydantic import PrivateAttr, SecretStr, ValidationError, model_validator
from typing_extensions import override

from distilabel.llm.base import AsyncLLM
from distilabel.utils.itertools import grouper

if TYPE_CHECKING:
    from huggingface_hub import AsyncInferenceClient
    from openai import AsyncOpenAI
    from transformers import PreTrainedTokenizer

    from distilabel.llm.typing import GenerateOutput
    from distilabel.steps.task.typing import ChatType


class InferenceEndpointsLLM(AsyncLLM):
    """InferenceEndpoints LLM implementation running the async API client via either
    the `huggingface_hub.AsyncInferenceClient` or via `openai.AsyncOpenAI`.

    Attributes:
        model_id: the model ID to use for the LLM as available in the Hugging Face Hub, which
            will be used to resolve the base URL for the serverless Inference Endpoints API requests.
            Defaults to `None`.
        endpoint_name: the name of the Inference Endpoint to use for the LLM. Defaults to `None`.
        endpoint_namespace: the namespace of the Inference Endpoint to use for the LLM. Defaults to `None`.
        base_url: the base URL to use for the Inference Endpoints API requests.
        api_key: the API key to authenticate the requests to the Inference Endpoints API.
        tokenizer_id: the tokenizer ID to use for the LLM as available in the Hugging Face Hub.
            Defaults to `None`, but defining one is recommended to properly format the prompt.
        model_display_name: the model display name to use for the LLM. Defaults to `None`.
        use_openai_client: whether to use the OpenAI client instead of the Hugging Face client.

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

    api_key: Optional[SecretStr] = os.getenv("HF_TOKEN", None)  # type: ignore

    tokenizer_id: Optional[str] = None
    model_display_name: Optional[str] = None
    use_openai_client: bool = False

    _model_name: Optional[str] = PrivateAttr(default=None)
    _tokenizer: Optional["PreTrainedTokenizer"] = PrivateAttr(default=None)
    _env_var: str = PrivateAttr(default="HF_TOKEN")
    _aclient: Optional[Union["AsyncInferenceClient", "AsyncOpenAI"]] = PrivateAttr(...)

    @model_validator(mode="after")
    def only_one_of_model_id_endpoint_name_or_base_url_provided(
        self,
    ) -> "InferenceEndpointsLLM":
        """Validates that only one of `model_id`, `endpoint_name`, or `base_url` is provided."""

        if self.model_id and (not self.endpoint_name and not self.base_url):
            return self
        if self.endpoint_name and (not self.model_id and not self.base_url):
            return self
        if self.base_url and (not self.model_id and not self.endpoint_name):
            return self

        raise ValidationError(
            "Only one of `model_id`, `endpoint_name`, or `base_url` must be provided. Found"
            f" `model_id`={self.model_id}, `endpoint_name`={self.endpoint_name}, and"
            f" `base_url`={self.base_url}."
        )

    def load(self, api_key: Optional[str] = None) -> None:
        """Loads the either the `AsyncInferenceClient` or the `AsyncOpenAI` client to benefit
        from async requests, running the Hugging Face Inference Endpoint underneath via the
        `/v1/chat/completions` endpoint, exposed for the models running on TGI using the
        `text-generation` task.

        Args:
            api_key: the API key to authenticate the requests to the Inference Endpoints API,
                which is the same as the Hugging Face Hub token.

        Raises:
            ImportError: if the `openai` Python client is not installed.
            ImportError: if the `huggingface-hub` Python client is not installed.
            ValueError: if the model is not currently deployed or is not running the TGI framework.
            ImportError: if the `transformers` Python client is not installed.
        """

        try:
            from huggingface_hub import (
                AsyncInferenceClient,
                InferenceClient,
                get_inference_endpoint,
            )
        except ImportError as ie:
            raise ImportError(
                "Hugging Face Hub Python client is not installed. Please install it using"
                " `pip install huggingface-hub`."
            ) from ie

        self.api_key = self._handle_api_key_value(
            self_value=self.api_key,
            load_value=api_key,
            env_var=self._env_var,
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
                client.resume().wait(timeout=300)

            self.base_url = client.url
            self._model_name = client.repository

        if self.use_openai_client:
            try:
                from openai import AsyncOpenAI
            except ImportError as ie:
                raise ImportError(
                    "OpenAI Python client is not installed. Please install it using"
                    " `pip install openai`."
                ) from ie

            self._aclient = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key.get_secret_value(),
                max_retries=6,
            )
        else:
            self._aclient = AsyncInferenceClient(
                model=self.base_url,
                token=self.api_key.get_secret_value(),
            )

        if self.tokenizer_id:
            try:
                from transformers import AutoTokenizer
            except ImportError as ie:
                raise ImportError(
                    "Transformers Python client is not installed. Please install it using"
                    " `pip install transformers`."
                ) from ie

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

    @property
    def model_name(self) -> Union[str, None]:
        """Returns the model name used for the LLM."""
        return (
            self.model_display_name
            or self._model_name
            or self.model_id
            or self.endpoint_name
            or self.base_url
        )

    async def _openai_agenerate(
        self,
        input: "ChatType",
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
    ) -> "GenerateOutput":
        """Generates completions for the given input using the OpenAI async client."""
        completion = await self._aclient.chat.completions.create(  # type: ignore
            messages=input,  # type: ignore
            model="tgi",
            max_tokens=max_new_tokens,
            n=1,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            timeout=50,
        )
        if completion.choices[0].message.content is None:
            self._logger.warning(
                f"⚠️ Received no response using OpenAI client (model: '{self.model_name}')."
                f" Finish reason was: {completion.choices[0].finish_reason}"
            )
        return [completion.choices[0].message.content]

    # TODO: add `num_generations` parameter once either TGI or `AsyncInferenceClient` allows `n` parameter
    async def agenerate(  # type: ignore
        self,
        input: "ChatType",
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: Optional[float] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
    ) -> "GenerateOutput":
        """Generates completions for the given input using the OpenAI async client.

        Args:
            input: a single input in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`, and only works with `1`.
            max_new_tokens: the maximun number of new tokens that the model will generate.
                Defaults to `128`.
            frequence_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`. Only applies if `use_openai_client=True`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`. Only applies if `use_openai_client=True`.
            repetition_penalty: the repetition penalty to use for the generation. Defaults
                to `None`. Only applies if `use_openai_client=False`.
            temperature: the temperature to use for the generation. Defaults to `1.0`.
            do_sample: whether to use sampling for the generation. Defaults to `False`.
                Only applies if `use_openai_client=False`.
            top_k: the top-k value to use for the generation. Defaults to `0.8`, since neither
                `0.0` nor `1.0` are valid values in TGI.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            typical_p: the typical-p value to use for the generation. Defaults to `0.5`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """

        if self.use_openai_client:
            return await self._openai_agenerate(
                input=input,
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                temperature=temperature,
                top_p=top_p,
            )

        if self._tokenizer is not None:
            prompt = self._tokenizer.apply_chat_template(  # type: ignore
                conversation=input,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join([message["content"] for message in input])

        try:
            completion = await self._aclient.text_generation(  # type: ignore
                prompt=prompt,  # type: ignore
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            return [completion]
        except Exception as e:
            self._logger.warning(
                f"⚠️ Received no response using Inference Client (model: '{self.model_name}')."
                f" Finish reason was: {e}"
            )
            return [None]

    # TODO: remove this function once `AsyncInferenceClient` allows `n` parameter
    @override
    def generate(
        self,
        inputs: List["ChatType"],
        num_generations: int = 1,
        **kwargs: Any,
    ) -> List["GenerateOutput"]:
        """Method to generate a list of responses asynchronously, returning the output
        synchronously awaiting for the response of each input sent to `agenerate`.
        """

        async def agenerate(
            inputs: List["ChatType"], **kwargs: Any
        ) -> "GenerateOutput":
            """Internal function to parallelize the asynchronous generation of responses."""
            tasks = [
                asyncio.create_task(self.agenerate(input=input, **kwargs))
                for input in inputs
                for _ in range(num_generations)
            ]
            return [outputs[0] for outputs in await asyncio.gather(*tasks)]

        outputs = self.event_loop.run_until_complete(agenerate(inputs, **kwargs))
        return list(grouper(outputs, n=num_generations, incomplete="ignore"))
