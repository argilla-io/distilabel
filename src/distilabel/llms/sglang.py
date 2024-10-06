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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from pydantic import Field, PrivateAttr, validate_call

from distilabel.llms.base import LLM
from distilabel.llms.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.llms.mixins.magpie import MagpieChatTemplateMixin
from distilabel.llms.typing import GenerateOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import FormattedInput, OutlinesStructuredOutputType

if TYPE_CHECKING:
    from sglang.srt.server import Runtime
    from transformers import PreTrainedTokenizer

    from distilabel.steps.tasks.typing import StandardInput

LogitsProcessorFn = Union[
    Callable[[List[int], Any], Any],
    Callable[[List[int], List[int], Any], Any],
]

LogitsProcessors = List[LogitsProcessorFn]


class SGLang(LLM, MagpieChatTemplateMixin, CudaDevicePlacementMixin):
    """`SGLang` library LLM implementation.

    Attributes:
        model (str): The model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        tokenizer_path (Optional[str]): Path to the tokenizer. If None, the default tokenizer for
            the model will be used.
        tokenizer_mode (str): Mode for tokenizer initialization. Default is "auto".
        skip_tokenizer_init (bool): Whether to skip tokenizer initialization. Default is False.
        load_format (str): Format for loading the model. Default is "auto".
        dtype (str): Data type for model parameters. Default is "auto".
        kv_cache_dtype (str): Data type for key-value cache. Default is "auto".
        trust_remote_code (bool): Whether to trust remote code when loading the model. Default is True.
        context_length (Optional[int]): Maximum context length for the model. If None, uses the
            model's default.
        quantization (Optional[str]): Quantization method to use. If None, no quantization is applied.
        served_model_name (Optional[str]): Name of the served model if using a model server.
        chat_template (Optional[str]): Custom chat template to use for formatting inputs.
        is_embedding (bool): Whether the model is used for embeddings. Default is False.

    Runtime parameters:
        - extra_kwargs: Additional dictionary of keyword arguments that will be passed to the
            SGLang class.
        - structured_output: The structured output format to use across all the generations.
        - log_level: The log level to use for the SGLang server.

    Examples:
        Generate text:

        ```python
        from distilabel.llms import SGLang

        llm = SGLang(model="your-model-name")
        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```
    """

    model: str
    dtype: str = "auto"
    trust_remote_code: bool = False
    quantization: Optional[str] = None

    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    chat_template: Optional[str] = None

    load_format: str = "auto"
    kv_cache_dtype: str = "auto"
    context_length: Optional[int] = None
    served_model_name: Optional[str] = None
    is_embedding: bool = False

    seed: int = 0

    extra_kwargs: Optional[RuntimeParameter[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Additional dictionary of keyword arguments that will be passed to the"
        " `SGLang` class.",
    )
    structured_output: Optional[RuntimeParameter[OutlinesStructuredOutputType]] = Field(
        default=None,
        description="The structured output format to use across all the generations.",
    )
    log_level: Optional[RuntimeParameter[str]] = Field(
        default="error",
        description="The log level to use for the SGLang server.",
    )

    _model: "Runtime" = PrivateAttr(None)
    _tokenizer: "PreTrainedTokenizer" = PrivateAttr(None)

    def load(self) -> None:
        """
        Loads the SGLang model using either path or Huggingface repository id.
        Additionally, this method also sets the `chat_template` for the tokenizer, so as to properly
        parse the list of OpenAI formatted inputs using the expected format by the model, otherwise, the
        default value is ChatML format, unless explicitly provided.
        """
        super().load()
        CudaDevicePlacementMixin.load(self)

        try:
            from sglang.srt.server import Runtime
        except ImportError as ie:
            raise ImportError(
                '`SGLang` is not installed. Please install it using `pip install "sglang[all]"`.'
                " Also, install FlashInfer CUDA kernels using:\n"
                "`pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/`"
            ) from ie

        self._model = Runtime(
            model_path=self.model,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization=self.quantization,
            tokenizer_path=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            skip_tokenizer_init=self.skip_tokenizer_init,
            load_format=self.load_format,
            kv_cache_dtype=self.kv_cache_dtype,
            context_length=self.context_length,
            served_model_name=self.served_model_name,
            is_embedding=self.is_embedding,
            random_seed=self.seed,
            **self.extra_kwargs,
        )

        self._tokenizer = self._model.get_tokenizer()  # type: ignore
        if self.chat_template is not None:
            self._tokenizer.chat_template = self.chat_template  # type: ignore

        if self.structured_output:
            self._structured_output_logits_processor = self._prepare_structured_output(
                self.structured_output
            )

    def unload(self) -> None:
        """Unloads the SGLang model."""
        self._model = None
        self._tokenizer = None
        CudaDevicePlacementMixin.unload(self)
        super().unload()

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.served_model_name

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
        from sglang.srt.sampling.sampling_params import SamplingParams

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
                max_new_tokens=max_new_tokens,
                stop=stop,
                stop_token_ids=stop_token_ids,
                include_stop_str_in_output=include_stop_str_in_output,
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


# You can add a ClientSGLang class here if needed, similar to ClientvLLM
