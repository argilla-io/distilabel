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
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from pydantic import Field, PrivateAttr, SecretStr, validate_call

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.llms.base import LLM
from distilabel.models.llms.openai import OpenAILLM
from distilabel.models.llms.typing import GenerateOutput
from distilabel.models.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.models.mixins.magpie import MagpieChatTemplateMixin
from distilabel.steps.tasks.typing import FormattedInput, OutlinesStructuredOutputType

if TYPE_CHECKING:
    from openai import OpenAI  # noqa
    from transformers import PreTrainedTokenizer
    from vllm import LLM as _vLLM

    from distilabel.steps.tasks.typing import StandardInput

LogitsProcessorFn = Union[
    Callable[[List[int], Any], Any],
    Callable[[List[int], List[int], Any], Any],
]

LogitsProcessors = List[LogitsProcessorFn]


class vLLM(LLM, MagpieChatTemplateMixin, CudaDevicePlacementMixin):
    """`vLLM` library LLM implementation.

    Attributes:
        model: the model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        dtype: the data type to use for the model. Defaults to `auto`.
        trust_remote_code: whether to trust the remote code when loading the model. Defaults
            to `False`.
        quantization: the quantization mode to use for the model. Defaults to `None`.
        revision: the revision of the model to load. Defaults to `None`.
        tokenizer: the tokenizer Hugging Face Hub repo id or a path to a directory containing
            the tokenizer files. If not provided, the tokenizer will be loaded from the
            model directory. Defaults to `None`.
        tokenizer_mode: the mode to use for the tokenizer. Defaults to `auto`.
        tokenizer_revision: the revision of the tokenizer to load. Defaults to `None`.
        skip_tokenizer_init: whether to skip the initialization of the tokenizer. Defaults
            to `False`.
        chat_template: a chat template that will be used to build the prompts before
            sending them to the model. If not provided, the chat template defined in the
            tokenizer config will be used. If not provided and the tokenizer doesn't have
            a chat template, then ChatML template will be used. Defaults to `None`.
        structured_output: a dictionary containing the structured output configuration or if more
            fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.
        seed: the seed to use for the random number generator. Defaults to `0`.
        extra_kwargs: additional dictionary of keyword arguments that will be passed to the
            `LLM` class of `vllm` library. Defaults to `{}`.
        _model: the `vLLM` model instance. This attribute is meant to be used internally
            and should not be accessed directly. It will be set in the `load` method.
        _tokenizer: the tokenizer instance used to format the prompt before passing it to
            the `LLM`. This attribute is meant to be used internally and should not be
            accessed directly. It will be set in the `load` method.
        use_magpie_template: a flag used to enable/disable applying the Magpie pre-query
            template. Defaults to `False`.
        magpie_pre_query_template: the pre-query template to be applied to the prompt or
            sent to the LLM to generate an instruction or a follow up user message. Valid
            values are "llama3", "qwen2" or another pre-query template provided. Defaults
            to `None`.

    References:
        - https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

    Runtime parameters:
        - `extra_kwargs`: additional dictionary of keyword arguments that will be passed to
            the `LLM` class of `vllm` library.

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import vLLM

        # You can pass a custom chat_template to the model
        llm = vLLM(
            model="prometheus-eval/prometheus-7b-v2.0",
            chat_template="[INST] {{ messages[0]\"content\" }}\\n{{ messages[1]\"content\" }}[/INST]",
        )

        llm.load()

        # Call the model
        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```

        Generate structured data:

        ```python
        from pathlib import Path
        from distilabel.models.llms import vLLM

        class User(BaseModel):
            name: str
            last_name: str
            id: int

        llm = vLLM(
            model="prometheus-eval/prometheus-7b-v2.0"
            structured_output={"format": "json", "schema": Character},
        )

        llm.load()

        # Call the model
        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
        ```
    """

    model: str
    dtype: str = "auto"
    trust_remote_code: bool = False
    quantization: Optional[str] = None
    revision: Optional[str] = None

    tokenizer: Optional[str] = None
    tokenizer_mode: Literal["auto", "slow"] = "auto"
    tokenizer_revision: Optional[str] = None
    skip_tokenizer_init: bool = False
    chat_template: Optional[str] = None

    seed: int = 0

    extra_kwargs: Optional[RuntimeParameter[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Additional dictionary of keyword arguments that will be passed to the"
        " `vLLM` class of `vllm` library. See all the supported arguments at: "
        "https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py",
    )
    structured_output: Optional[RuntimeParameter[OutlinesStructuredOutputType]] = Field(
        default=None,
        description="The structured output format to use across all the generations.",
    )

    _model: "_vLLM" = PrivateAttr(None)
    _tokenizer: "PreTrainedTokenizer" = PrivateAttr(None)
    _structured_output_logits_processor: Optional[Callable] = PrivateAttr(default=None)

    def load(self) -> None:
        """Loads the `vLLM` model using either the path or the Hugging Face Hub repository id.
        Additionally, this method also sets the `chat_template` for the tokenizer, so as to properly
        parse the list of OpenAI formatted inputs using the expected format by the model, otherwise, the
        default value is ChatML format, unless explicitly provided.
        """
        super().load()

        CudaDevicePlacementMixin.load(self)

        try:
            from vllm import LLM as _vLLM
        except ImportError as ie:
            raise ImportError(
                "vLLM is not installed. Please install it using `pip install vllm`."
            ) from ie

        self._model = _vLLM(
            self.model,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization=self.quantization,
            revision=self.revision,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            tokenizer_revision=self.tokenizer_revision,
            skip_tokenizer_init=self.skip_tokenizer_init,
            seed=self.seed,
            **self.extra_kwargs,  # type: ignore
        )

        self._tokenizer = self._model.get_tokenizer()  # type: ignore
        if self.chat_template is not None:
            self._tokenizer.chat_template = self.chat_template  # type: ignore

        if self.structured_output:
            self._structured_output_logits_processor = self._prepare_structured_output(
                self.structured_output
            )

    def unload(self) -> None:
        """Unloads the `vLLM` model."""
        self._model = None  # type: ignore
        self._tokenizer = None  # type: ignore
        CudaDevicePlacementMixin.unload(self)
        super().unload()

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    def prepare_input(self, input: "StandardInput") -> str:
        """Prepares the input (applying the chat template and tokenization) for the provided
        input.

        Args:
            input: the input list containing chat items.

        Returns:
            The prompt to send to the LLM.
        """
        if self._tokenizer.chat_template is None:
            return input[0]["content"]

        prompt: str = (
            self._tokenizer.apply_chat_template(
                input,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,  # type: ignore
            )
            if input
            else ""
        )
        return super().apply_magpie_pre_query_template(prompt, input)

    def _prepare_batches(
        self, inputs: List[FormattedInput]
    ) -> Tuple[List[List[FormattedInput]], List[int]]:
        """Prepares the inputs by grouping them by the structured output.

        When we generate structured outputs with schemas obtained from a dataset, we need to
        prepare the data to try to send batches of inputs instead of single inputs to the model
        to take advante of the engine. So we group the inputs by the structured output to be
        passed in the `generate` method.

        Args:
            inputs: The batch of inputs passed to the generate method. As we expect to be generating
                structured outputs, each element will be a tuple containing the instruction and the
                structured output.

        Returns:
            The prepared batches (sub-batches let's say) to be passed to the `generate` method.
            Each new tuple will contain instead of the single instruction, a list of instructions
        """
        instruction_order = {}
        batches = {}
        for i, (instruction, structured_output) in enumerate(inputs):
            instruction = self.prepare_input(instruction)
            instruction_order[instruction] = i
            structured_output = json.dumps(structured_output)
            if structured_output not in batches:
                batches[structured_output] = [instruction]
            else:
                batches[structured_output].append(instruction)

        # Flatten the instructions in prepared_data
        flat_instructions = [
            instruction for _, group in batches.items() for instruction in group
        ]
        # Generate the list of indices based on the original order
        sorted_indices = [
            instruction_order[instruction] for instruction in flat_instructions
        ]
        return [
            (batch, json.loads(schema)) for schema, batch in batches.items()
        ], sorted_indices

    @validate_call
    def generate(  # type: ignore
        self,
        inputs: List[FormattedInput],
        num_generations: int = 1,
        max_new_tokens: int = 128,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        include_stop_str_in_output: bool = False,
        logits_processors: Optional[LogitsProcessors] = None,
        extra_sampling_params: Optional[Dict[str, Any]] = None,
    ) -> List[GenerateOutput]:
        """Generates `num_generations` responses for each input.

        Args:
            inputs: a list of inputs in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`.
            frequency_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            repetition_penalty: the repetition penalty to use for the generation Defaults to
                `1.0`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            top_k: the top-k value to use for the generation. Defaults to `0`.
            min_p: the minimum probability to use for the generation. Defaults to `0.0`.
            stop: a list of strings that will be used to stop the generation when found.
                Defaults to `None`.
            stop_token_ids: a list of token ids that will be used to stop the generation
                when found. Defaults to `None`.
            include_stop_str_in_output: whether to include the stop string in the output.
                Defaults to `False`.
            logits_processors: a list of functions to process the logits before sampling.
                Defaults to `None`.
            extra_sampling_params: dictionary with additional arguments to be passed to
                the `SamplingParams` class from `vllm`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        from vllm import SamplingParams

        if not logits_processors:
            logits_processors = []

        if extra_sampling_params is None:
            extra_sampling_params = {}

        structured_output = None

        if isinstance(inputs[0], tuple):
            prepared_batches, sorted_indices = self._prepare_batches(inputs)
        else:
            # Simulate a batch without the structured output content
            prepared_batches = [([self.prepare_input(input) for input in inputs], None)]
            sorted_indices = None

        # Case in which we have a single structured output for the dataset
        if self._structured_output_logits_processor:
            logits_processors.append(self._structured_output_logits_processor)

        batched_outputs = []

        for prepared_inputs, structured_output in prepared_batches:
            if structured_output:
                logits_processors.append(
                    self._prepare_structured_output(structured_output)
                )

            sampling_params = SamplingParams(  # type: ignore
                n=num_generations,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_new_tokens,
                stop=stop,
                stop_token_ids=stop_token_ids,
                include_stop_str_in_output=include_stop_str_in_output,
                logits_processors=logits_processors,
                **extra_sampling_params,
            )

            batch_outputs = self._model.generate(
                prepared_inputs,
                sampling_params,
                use_tqdm=False,  # type: ignore
            )

            batched_outputs += [
                [output.text for output in outputs.outputs] for outputs in batch_outputs
            ]

        # If logits_processor is set, we need to sort the outputs back to the original order
        # (would be needed only if we have multiple structured outputs in the dataset)
        if sorted_indices is not None:
            batched_outputs = _sort_batches(
                batched_outputs, sorted_indices, num_generations=num_generations
            )
        return batched_outputs

    def _prepare_structured_output(
        self, structured_output: Optional[OutlinesStructuredOutputType] = None
    ) -> Union[Callable, None]:
        """Creates the appropriate function to filter tokens to generate structured outputs.

        Args:
            structured_output: the configuration dict to prepare the structured output.

        Returns:
            The callable that will be used to guide the generation of the model.
        """
        from distilabel.steps.tasks.structured_outputs.outlines import (
            prepare_guided_output,
        )

        result = prepare_guided_output(structured_output, "vllm", self._model)
        if (schema := result.get("schema")) and self.structured_output:
            self.structured_output["schema"] = schema
        return result["processor"]


class ClientvLLM(OpenAILLM, MagpieChatTemplateMixin):
    """A client for the `vLLM` server implementing the OpenAI API specification.

    Attributes:
        base_url: the base URL of the `vLLM` server. Defaults to `"http://localhost:8000"`.
        max_retries: the maximum number of times to retry the request to the API before
            failing. Defaults to `6`.
        timeout: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.
        httpx_client_kwargs: extra kwargs that will be passed to the `httpx.AsyncClient`
            created to comunicate with the `vLLM` server. Defaults to `None`.
        tokenizer: the Hugging Face Hub repo id or path of the tokenizer that will be used
            to apply the chat template and tokenize the inputs before sending it to the
            server. Defaults to `None`.
        tokenizer_revision: the revision of the tokenizer to load. Defaults to `None`.
        _aclient: the `httpx.AsyncClient` used to comunicate with the `vLLM` server. Defaults
            to `None`.

    Runtime parameters:
        - `base_url`: the base url of the `vLLM` server. Defaults to `"http://localhost:8000"`.
        - `max_retries`: the maximum number of times to retry the request to the API before
            failing. Defaults to `6`.
        - `timeout`: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.
        - `httpx_client_kwargs`: extra kwargs that will be passed to the `httpx.AsyncClient`
            created to comunicate with the `vLLM` server. Defaults to `None`.

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import ClientvLLM

        llm = ClientvLLM(
            base_url="http://localhost:8000/v1",
            tokenizer="meta-llama/Meta-Llama-3.1-8B-Instruct"
        )

        llm.load()

        results = llm.generate_outputs(
            inputs=[[{"role": "user", "content": "Hello, how are you?"}]],
            temperature=0.7,
            top_p=1.0,
            max_new_tokens=256,
        )
        # [
        #     [
        #         "I'm functioning properly, thank you for asking. How can I assist you today?",
        #         "I'm doing well, thank you for asking. I'm a large language model, so I don't have feelings or emotions like humans do, but I'm here to help answer any questions or provide information you might need. How can I assist you today?",
        #         "I'm just a computer program, so I don't have feelings like humans do, but I'm functioning properly and ready to help you with any questions or tasks you have. What's on your mind?"
        #     ]
        # ]
        ```
    """

    model: str = ""  # Default value so it's not needed to `ClientvLLM(model="...")`
    tokenizer: Optional[str] = None
    tokenizer_revision: Optional[str] = None

    # We need the sync client to get the list of models
    _client: "OpenAI" = PrivateAttr(None)
    _tokenizer: "PreTrainedTokenizer" = PrivateAttr(None)

    def load(self) -> None:
        """Creates an `httpx.AsyncClient` to connect to the vLLM server and a tokenizer
        optionally."""

        self.api_key = SecretStr("EMPTY")

        # We need to first create the sync client to get the model name that will be used
        # in the `super().load()` when creating the logger.
        try:
            from openai import OpenAI
        except ImportError as ie:
            raise ImportError(
                "OpenAI Python client is not installed. Please install it using"
                " `pip install openai`."
            ) from ie

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key.get_secret_value(),  # type: ignore
            max_retries=self.max_retries,  # type: ignore
            timeout=self.timeout,
        )

        super().load()

        try:
            from transformers import AutoTokenizer
        except ImportError as ie:
            raise ImportError(
                "To use `ClientvLLM` you need to install `transformers`."
                "Please install it using `pip install transformers`."
            ) from ie

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer, revision=self.tokenizer_revision
        )

    @cached_property
    def model_name(self) -> str:  # type: ignore
        """Returns the name of the model served with vLLM server."""
        models = self._client.models.list()
        return models.data[0].id

    def _prepare_input(self, input: "StandardInput") -> str:
        """Prepares the input (applying the chat template and tokenization) for the provided
        input.

        Args:
            input: the input list containing chat items.

        Returns:
            The prompt to send to the LLM.
        """
        prompt: str = (
            self._tokenizer.apply_chat_template(  # type: ignore
                input,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,  # type: ignore
            )
            if input
            else ""
        )
        return super().apply_magpie_pre_query_template(prompt, input)

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: FormattedInput,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[Dict[str, int]] = None,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> GenerateOutput:
        """Generates `num_generations` responses for each input.

        Args:
            input: a single input in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            frequency_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            logit_bias: modify the likelihood of specified tokens appearing in the completion.
                Defaults to ``
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: nucleus sampling. The value refers to the top-p tokens that should be
                considered for sampling. Defaults to `1.0`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """

        completion = await self._aclient.completions.create(
            model=self.model_name,
            prompt=self._prepare_input(input),  # type: ignore
            n=num_generations,
            max_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
        )

        generations = []
        for choice in completion.choices:
            if (text := choice.text) == "":
                self._logger.warning(  # type: ignore
                    f"Received no response from vLLM server (model: '{self.model_name}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(text)
        return generations


def _sort_batches(
    batches: List[List[FormattedInput]], indices: List[int], num_generations: int = 1
) -> List[str]:
    """Helper function to sort back the mini-batches generated by the model.

    It must take into account the number of `num_generations` to repeat the indices
    accordingly.

    Args:
        batches: The mini-batches generated by the model.
        indices: The indices that would sort the mini-batches back to the original order.
        num_generations: The number of generations requested to vLLM. Defaults to 1.

    Returns:
        Sorted batched_outputs.
    """
    batch_sizes = [len(batch) for batch in batches]
    flattened_batches = np.array([b for batch in batches for b in batch])
    sorted_batches = np.take_along_axis(
        flattened_batches,
        np.argsort(np.repeat(indices, num_generations)),
        axis=0,
    ).tolist()
    sorted_batches = _batchify(sorted_batches, batch_sizes)
    return sorted_batches


def _batchify(sorted_batches: List[str], batch_sizes: List[int]) -> List[List[str]]:
    """Helper function to regenerate the sorted batches from the flattened sorted ones.

    Args:
        sorted_batches: Output obtained from the `_sort_batches` function.
        batch_sizes: The batch sizes to be used to split the sorted batches.

    Returns:
        Batched sorted batches in the original shape.
    """
    batches = []
    idx = 0
    for bs in batch_sizes:
        batches.append(sorted_batches[idx : idx + bs])
        idx += bs
    return batches
