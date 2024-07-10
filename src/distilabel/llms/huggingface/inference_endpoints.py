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
import random
import sys
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import (
    Field,
    PrivateAttr,
    SecretStr,
    ValidationError,
    model_validator,
    validate_call,
)
from typing_extensions import override

from distilabel.llms.base import AsyncLLM
from distilabel.llms.mixins.magpie import MagpieChatTemplateMixin
from distilabel.llms.typing import GenerateOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import (
    FormattedInput,
    StandardInput,
    StructuredOutputType,
)
from distilabel.utils.huggingface import (
    _INFERENCE_ENDPOINTS_API_KEY_ENV_VAR_NAME,
    get_hf_token,
)

if TYPE_CHECKING:
    from huggingface_hub import AsyncInferenceClient
    from transformers import PreTrainedTokenizer


class InferenceEndpointsLLM(AsyncLLM, MagpieChatTemplateMixin):
    """InferenceEndpoints LLM implementation running the async API client.

    This LLM will internally use `huggingface_hub.AsyncInferenceClient`.

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

    Icon:
        `:hugging:`

    Examples:

        Free serverless Inference API:

        ```python
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        llm = InferenceEndpointsLLM(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
        )

        llm.load()

        output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```

        Dedicated Inference Endpoints:

        ```python
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        llm = InferenceEndpointsLLM(
            endpoint_name="<ENDPOINT_NAME>",
            api_key="<HF_API_KEY>",
            endpoint_namespace="<USER|ORG>",
        )

        llm.load()

        output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```

        Dedicated Inference Endpoints or TGI:

        ```python
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        llm = InferenceEndpointsLLM(
            api_key="<HF_API_KEY>",
            base_url="<BASE_URL>",
        )

        llm.load()

        output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```
    """

    model_id: Optional[str] = None

    endpoint_name: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The name of the Inference Endpoint to use for the LLM.",
    )
    endpoint_namespace: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The namespace of the Inference Endpoint to use for the LLM.",
    )
    base_url: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The base URL to use for the Inference Endpoints API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default=os.getenv(_INFERENCE_ENDPOINTS_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Inference Endpoints API.",
    )

    tokenizer_id: Optional[str] = None
    model_display_name: Optional[str] = None

    structured_output: Optional[RuntimeParameter[StructuredOutputType]] = Field(
        default=None,
        description="The structured output format to use across all the generations.",
    )

    _num_generations_param_supported = False

    _model_name: Optional[str] = PrivateAttr(default=None)
    _tokenizer: Optional["PreTrainedTokenizer"] = PrivateAttr(default=None)
    _api_key_env_var: str = PrivateAttr(_INFERENCE_ENDPOINTS_API_KEY_ENV_VAR_NAME)
    _aclient: Optional["AsyncInferenceClient"] = PrivateAttr(...)

    @model_validator(mode="after")  # type: ignore
    def only_one_of_model_id_endpoint_name_or_base_url_provided(
        self,
    ) -> "InferenceEndpointsLLM":
        """Validates that only one of `model_id` or `endpoint_name` is provided; and if `base_url` is also
        provided, a warning will be shown informing the user that the provided `base_url` will be ignored in
        favour of the dynamically calculated one.."""

        if self.base_url and (self.model_id or self.endpoint_name):
            self._logger.warning(  # type: ignore
                f"Since the `base_url={self.base_url}` is available and either one of `model_id`"
                " or `endpoint_name` is also provided, the `base_url` will either be ignored"
                " or overwritten with the one generated from either of those args, for serverless"
                " or dedicated inference endpoints, respectively."
            )

        if self.model_id and self.tokenizer_id is None:
            self.tokenizer_id = self.model_id

        if self.use_magpie_template and self.tokenizer_id is None:
            raise ValueError(
                "`use_magpie_template` cannot be `True` if `tokenizer_id` is `None`. Please,"
                " set a `tokenizer_id` and try again."
            )

        if self.base_url and not (self.model_id or self.endpoint_name):
            return self

        if self.model_id and not self.endpoint_name:
            return self

        if self.endpoint_name and not self.model_id:
            return self

        raise ValidationError(
            f"Only one of `model_id` or `endpoint_name` must be provided. If `base_url` is"
            f" provided too, it will be overwritten instead. Found `model_id`={self.model_id},"
            f" `endpoint_name`={self.endpoint_name}, and `base_url`={self.base_url}."
        )

    def load(self) -> None:  # noqa: C901
        """Loads the `AsyncInferenceClient` client to connect to the Hugging Face Inference
        Endpoint.

        Raises:
            ImportError: if the `huggingface-hub` Python client is not installed.
            ValueError: if the model is not currently deployed or is not running the TGI framework.
            ImportError: if the `transformers` Python client is not installed.
        """
        super().load()

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

        if self.api_key is None:
            self.api_key = SecretStr(get_hf_token(self.__class__.__name__, "api_key"))

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
            elif client.status == "initializing":
                client.wait(timeout=300)

            self.base_url = client.url
            self._model_name = client.repository

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
    @override
    def model_name(self) -> Union[str, None]:  # type: ignore
        """Returns the model name used for the LLM."""
        return (
            self.model_display_name
            or self._model_name
            or self.model_id
            or self.endpoint_name
            or self.base_url
        )

    def prepare_input(self, input: "StandardInput") -> str:
        """Prepares the input (applying the chat template and tokenization) for the provided
        input.

        Args:
            input: the input list containing chat items.

        Returns:
            The prompt to send to the LLM.
        """
        prompt: str = self._tokenizer.apply_chat_template(  # type: ignore
            conversation=input,  # type: ignore
            tokenize=False,
            add_generation_prompt=True,
        )
        return super().apply_magpie_pre_query_template(prompt, input)

    def get_structured_output(
        self, input: FormattedInput
    ) -> Union[Dict[str, Any], None]:
        """Gets the structured output (if any) for the given input.

        Args:
            input: a single input in chat format to generate responses for.

        Returns:
            The structured output that will be passed as `grammer` to the inference endpoint
            or `None` if not required.
        """
        structured_output = None

        # Specific structured output per input
        if isinstance(input, tuple):
            input, structured_output = input
            structured_output = {
                "type": structured_output["format"],
                "value": structured_output["schema"],
            }

        # Same structured output for all the inputs
        if structured_output is None and self.structured_output is not None:
            try:
                structured_output = {
                    "type": self.structured_output["format"],
                    "value": self.structured_output["schema"],
                }
            except KeyError as e:
                raise ValueError(
                    "To use the structured output you have to inform the `format` and `schema` in "
                    "the `structured_output` attribute."
                ) from e

        return structured_output

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: FormattedInput,
        max_new_tokens: int = 128,
        repetition_penalty: Optional[float] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        stop_sequences: Optional[Union[str, List[str]]] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        watermark: bool = False,
    ) -> GenerateOutput:
        """Generates completions for the given input using the async client.

        Args:
            input: a single input in chat format to generate responses for.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            repetition_penalty: the repetition penalty to use for the generation. Defaults
                to `None`.
            temperature: the temperature to use for the generation. Defaults to `1.0`.
            do_sample: whether to use sampling for the generation. Defaults to `False`.
            top_k: the top-k value to use for the generation. Defaults to `0.8`, since neither
                `0.0` nor `1.0` are valid values in TGI.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            typical_p: the typical-p value to use for the generation. Defaults to `0.5`.
            stop_sequences: either a single string or a list of strings containing the sequences
                to stop the generation at. Defaults to `None`, but will be set to the
                `tokenizer.eos_token` if available.
            return_full_text: whether to return the full text of the completion or just the
                generated text. Defaults to `False`, meaning that only the generated text will be
                returned.
            seed: the seed to use for the generation. Defaults to `None`.
            watermark: whether to add the watermark to the generated text. Defaults to `None`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        if stop_sequences is not None:
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            if len(stop_sequences) > 4:
                warnings.warn(
                    "Only up to 4 stop sequences are allowed, so keeping the first 4 items only.",
                    UserWarning,
                    stacklevel=2,
                )
                stop_sequences = stop_sequences[:4]

        structured_output = self.get_structured_output(input)

        completion = None
        try:
            completion = await self._aclient.text_generation(  # type: ignore
                prompt=self.prepare_input(input),  # type: ignore
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
                return_full_text=return_full_text,
                watermark=watermark,
                grammar=structured_output,  # type: ignore
                # NOTE: here to ensure that the cache is not used and a different response is
                # generated every time
                seed=seed or random.randint(0, sys.maxsize),
            )
        except Exception as e:
            self._logger.warning(  # type: ignore
                f"⚠️ Received no response using Inference Client (model: '{self.model_name}')."
                f" Finish reason was: {e}"
            )

        return [completion]
