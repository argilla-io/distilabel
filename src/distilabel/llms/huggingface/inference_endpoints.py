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
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    Field,
    PrivateAttr,
    SecretStr,
    ValidationError,
    model_validator,
    validate_call,
)
from pydantic._internal._model_construction import ModelMetaclass
from typing_extensions import Annotated, override

from distilabel.llms.base import AsyncLLM
from distilabel.llms.mixins.magpie import MagpieChatTemplateMixin
from distilabel.llms.typing import GenerateOutput
from distilabel.llms.utils import prepare_output
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import (
    FormattedInput,
    StandardInput,
    StructuredOutputType,
)
from distilabel.utils.huggingface import HF_TOKEN_ENV_VAR, get_hf_token

if TYPE_CHECKING:
    from huggingface_hub import AsyncInferenceClient
    from huggingface_hub.inference._generated.types.chat_completion import (
        ChatCompletionOutput,
    )
    from huggingface_hub.inference._generated.types.text_generation import (
        TextGenerationOutput,
    )
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
        use_magpie_template: a flag used to enable/disable applying the Magpie pre-query
            template. Defaults to `False`.
        magpie_pre_query_template: the pre-query template to be applied to the prompt or
            sent to the LLM to generate an instruction or a follow up user message. Valid
            values are "llama3", "qwen2" or another pre-query template provided. Defaults
            to `None`.
        structured_output: a dictionary containing the structured output configuration or
            if more fine-grained control is needed, an instance of `OutlinesStructuredOutput`.
            Defaults to None.

    Icon:
        `:hugging:`

    Examples:
        Free serverless Inference API, set the input_batch_size of the Task that uses this to avoid Model is overloaded:

        ```python
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        llm = InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        )

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
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

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```

        Dedicated Inference Endpoints or TGI:

        ```python
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        llm = InferenceEndpointsLLM(
            api_key="<HF_API_KEY>",
            base_url="<BASE_URL>",
        )

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```

        Generate structured data:

        ```python
        from pydantic import BaseModel
        from distilabel.llms import InferenceEndpointsLLM

        class User(BaseModel):
            name: str
            last_name: str
            id: int

        llm = InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
            api_key="api.key",
            structured_output={"format": "json", "schema": User.model_json_schema()}
        )

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the Tour De France"}]])
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
        default_factory=lambda: os.getenv(HF_TOKEN_ENV_VAR),
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
    _api_key_env_var: str = PrivateAttr(HF_TOKEN_ENV_VAR)
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

        if self.use_magpie_template and self.tokenizer_id is None:
            raise ValueError(
                "`use_magpie_template` cannot be `True` if `tokenizer_id` is `None`. Please,"
                " set a `tokenizer_id` and try again."
            )

        if (
            self.model_id
            and self.tokenizer_id is None
            and self.structured_output is not None
        ):
            self.tokenizer_id = self.model_id

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
            client = InferenceClient(
                model=self.model_id, token=self.api_key.get_secret_value()
            )
            status = client.get_model_status()

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
            base_url=self.base_url,
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
        prompt: str = (
            self._tokenizer.apply_chat_template(  # type: ignore
                conversation=input,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,
            )
            if input
            else ""
        )
        return super().apply_magpie_pre_query_template(prompt, input)

    def _get_structured_output(
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
                "type": structured_output["format"],  # type: ignore
                "value": structured_output["schema"],  # type: ignore
            }

        # Same structured output for all the inputs
        if structured_output is None and self.structured_output is not None:
            try:
                structured_output = {
                    "type": self.structured_output["format"],  # type: ignore
                    "value": self.structured_output["schema"],  # type: ignore
                }
            except KeyError as e:
                raise ValueError(
                    "To use the structured output you have to inform the `format` and `schema` in "
                    "the `structured_output` attribute."
                ) from e

        if structured_output:
            if isinstance(structured_output["value"], ModelMetaclass):
                structured_output["value"] = structured_output[
                    "value"
                ].model_json_schema()

        return structured_output

    async def _generate_with_text_generation(
        self,
        input: FormattedInput,
        max_new_tokens: int = 128,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        stop_sequences: Union[List[str], None] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        watermark: bool = False,
    ) -> GenerateOutput:
        structured_output = self._get_structured_output(input)

        completion = None
        try:
            completion: "TextGenerationOutput" = await self._aclient.text_generation(  # type: ignore
                prompt=self.prepare_input(input),  # type: ignore
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
                return_full_text=return_full_text,
                # NOTE: here to ensure that the cache is not used and a different response is
                # generated every time
                seed=seed or random.randint(0, sys.maxsize),
                watermark=watermark,
                grammar=structured_output,  # type: ignore
                details=True,
            )
        except Exception as e:
            self._logger.warning(  # type: ignore
                f"⚠️ Received no response using Inference Client (model: '{self.model_name}')."
                f" Finish reason was: {e}"
            )
        # NOTE: I cannot see the input tokens returned, and given that the model can be private, I cannot
        # count them...
        return prepare_output(
            [completion.generated_text],
            input_tokens=[0],
            output_tokens=[
                completion.details.generated_tokens if completion.details else 0
            ],
        )

    async def _generate_with_chat_completion(
        self,
        input: "StandardInput",
        max_new_tokens: int = 128,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 1.0,
        tool_choice: Optional[Union[Dict[str, str], Literal["auto"]]] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_p: Optional[float] = None,
    ) -> GenerateOutput:
        message = None
        try:
            completion: "ChatCompletionOutput" = await self._aclient.chat_completion(  # type: ignore
                messages=input,  # type: ignore
                max_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                presence_penalty=presence_penalty,
                # NOTE: here to ensure that the cache is not used and a different response is
                # generated every time
                seed=seed or random.randint(0, sys.maxsize),
                stop=stop_sequences,
                temperature=temperature,
                tool_choice=tool_choice,  # type: ignore
                tool_prompt=tool_prompt,
                tools=tools,  # type: ignore
                top_p=top_p,
            )
            choice = completion.choices[0]
            if (message := choice.message.content) is None:
                self._logger.warning(  # type: ignore
                    f"⚠️ Received no response using Inference Client (model: '{self.model_name}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
        except Exception as e:
            self._logger.warning(  # type: ignore
                f"⚠️ Received no response using Inference Client (model: '{self.model_name}')."
                f" Finish reason was: {e}"
            )
        return prepare_output(
            [message],
            input_tokens=[completion.usage.prompt_tokens],
            output_tokens=[completion.usage.completion_tokens],
        )

    def _check_stop_sequences(
        self,
        stop_sequences: Optional[Union[str, List[str]]] = None,
    ) -> Union[List[str], None]:
        """Checks that no more than 4 stop sequences are provided.

        Args:
            stop_sequences: the stop sequences to be checked.

        Returns:
            The stop sequences.
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
        return stop_sequences

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: FormattedInput,
        max_new_tokens: int = 128,
        frequency_penalty: Optional[Annotated[float, Field(ge=-2.0, le=2.0)]] = None,
        logit_bias: Optional[List[float]] = None,
        presence_penalty: Optional[Annotated[float, Field(ge=-2.0, le=2.0)]] = None,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 1.0,
        tool_choice: Optional[Union[Dict[str, str], Literal["auto"]]] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        top_k: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
    ) -> GenerateOutput:
        """Generates completions for the given input using the async client. This method
        uses two methods of the `huggingface_hub.AsyncClient`: `chat_completion` and `text_generation`.
        `chat_completion` method will be used only if no `tokenizer_id` has been specified.
        Some arguments of this function are specific to the `text_generation` method, while
        some others are specific to the `chat_completion` method.

        Args:
            input: a single input in chat format to generate responses for.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            frequency_penalty: a value between `-2.0` and `2.0`. Positive values penalize
                new tokens based on their existing frequency in the text so far, decreasing
                model's likelihood to repeat the same line verbatim. Defauls to `None`.
            logit_bias: modify the likelihood of specified tokens appearing in the completion.
                This argument is exclusive to the `chat_completion` method and will be used
                only if `tokenizer_id` is `None`.
                Defaults to `None`.
            presence_penalty: a value between `-2.0` and `2.0`. Positive values penalize
                new tokens based on whether they appear in the text so far, increasing the
                model likelihood to talk about new topics. This argument is exclusive to
                the `chat_completion` method and will be used only if `tokenizer_id` is
                `None`. Defauls to `None`.
            seed: the seed to use for the generation. Defaults to `None`.
            stop_sequences: either a single string or a list of strings containing the sequences
                to stop the generation at. Defaults to `None`, but will be set to the
                `tokenizer.eos_token` if available.
            temperature: the temperature to use for the generation. Defaults to `1.0`.
            tool_choice: the name of the tool the model should call. It can be a dictionary
                like `{"function_name": "my_tool"}` or "auto". If not provided, then the
                model won't use any tool. This argument is exclusive to the `chat_completion`
                method and will be used only if `tokenizer_id` is `None`. Defaults to `None`.
            tool_prompt: A prompt to be appended before the tools. This argument is exclusive
                to the `chat_completion` method and will be used only if `tokenizer_id`
                is `None`. Defauls to `None`.
            tools: a list of tools definitions that the LLM can use.
                This argument is exclusive to the `chat_completion` method and will be used
                only if `tokenizer_id` is `None`. Defaults to `None`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            do_sample: whether to use sampling for the generation. This argument is exclusive
                of the `text_generation` method and will be only used if `tokenizer_id` is not
                `None`. Defaults to `False`.
            repetition_penalty: the repetition penalty to use for the generation. This argument
                is exclusive of the `text_generation` method and will be only used if `tokenizer_id`
                is not `None`. Defaults to `None`.
            return_full_text: whether to return the full text of the completion or just
                the generated text. Defaults to `False`, meaning that only the generated
                text will be returned. This argument is exclusive of the `text_generation`
                method and will be only used if `tokenizer_id` is not `None`.
            top_k: the top-k value to use for the generation. This argument is exclusive
                of the `text_generation` method and will be only used if `tokenizer_id`
                is not `None`. Defaults to `0.8`, since neither `0.0` nor `1.0` are valid
                values in TGI.
            typical_p: the typical-p value to use for the generation. This argument is exclusive
                of the `text_generation` method and will be only used if `tokenizer_id`
                is not `None`. Defaults to `None`.
            watermark: whether to add the watermark to the generated text. This argument
                is exclusive of the `text_generation` method and will be only used if `tokenizer_id`
                is not `None`. Defaults to `None`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        stop_sequences = self._check_stop_sequences(stop_sequences)

        if self.tokenizer_id is None:
            return await self._generate_with_chat_completion(
                input=input,  # type: ignore
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                presence_penalty=presence_penalty,
                seed=seed,
                stop_sequences=stop_sequences,
                temperature=temperature,
                tool_choice=tool_choice,
                tool_prompt=tool_prompt,
                tools=tools,
                top_p=top_p,
            )

        return await self._generate_with_text_generation(
            input=input,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            typical_p=typical_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            return_full_text=return_full_text,
            seed=seed,
            watermark=watermark,
        )
