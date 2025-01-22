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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from pydantic import (
    Field,
    PrivateAttr,
    validate_call,
)

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.llms.base import LLM
from distilabel.models.llms.utils import compute_tokens, prepare_output
from distilabel.models.mixins.magpie import MagpieChatTemplateMixin
from distilabel.typing import (
    StandardInput,
    GenerateOutput,
    OutlinesStructuredOutputType,
)

if TYPE_CHECKING:
    import mlx.nn as nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper


class MlxModel:
    """Wrapper class providing a consistent interface for MLX models."""

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer


class MlxLLM(LLM, MagpieChatTemplateMixin):
    """Apple MLX LLM implementation.

    Attributes:
        path_or_hf_repo: the path to the model or the Hugging Face Hub repo id.
        tokenizer_config: the tokenizer configuration.
        mlx_model_config: the MLX model configuration.
        adapter_path: the path to the adapter.
        use_magpie_template: a flag used to enable/disable applying the Magpie pre-query
            template. Defaults to `False`.
        magpie_pre_query_template: the pre-query template to be applied to the prompt or
            sent to the LLM to generate an instruction or a follow up user message. Valid
            values are "llama3", "qwen2" or another pre-query template provided. Defaults
            to `None`.

    Icon:
        `:apple:`

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import MlxLLM

        llm = MlxLLM(path_or_hf_repo="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

        llm.load()

        # Call the model
        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```
    """

    path_or_hf_repo: str
    tokenizer_config: Dict[str, Any] = Field(default_factory=dict)
    mlx_model_config: Dict[str, Any] = Field(default_factory=dict)
    adapter_path: Optional[str] = None
    structured_output: Optional[RuntimeParameter[OutlinesStructuredOutputType]] = Field(
        default=None,
        description="The structured output format to use across all the generations.",
    )
    _model: Optional["nn.Module"] = PrivateAttr(None)
    _tokenizer: Optional["TokenizerWrapper"] = PrivateAttr(None)
    _wrapped_model: Optional[Any] = PrivateAttr(None)
    _mlx_generate: Optional[Callable] = PrivateAttr(None)
    _make_sampler: Optional[Callable] = PrivateAttr(None)

    def load(self) -> None:
        """Loads the model and tokenizer and creates the text generation pipeline. In addition,
        it will configure the tokenizer chat template."""
        try:
            import mlx  # noqa
            from mlx_lm.utils import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as ie:
            raise ImportError(
                "MLX is not installed. Please install it using `pip install 'distilabel[mlx]'`."
            ) from ie

        self._model, self._tokenizer = load(
            self.path_or_hf_repo,
            tokenizer_config=self.tokenizer_config,
            model_config=self.mlx_model_config,
            adapter_path=self.adapter_path,
        )
        self._wrapped_model = MlxModel(self._model, self._tokenizer)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._mlx_generate = generate
        self._make_sampler = make_sampler
        super().load()

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.path_or_hf_repo

    def prepare_input(self, input: Union["StandardInput", str]) -> str:
        """Prepares the input (applying the chat template and tokenization) for the provided
        input.

        Args:
            input: the input list containing chat items.

        Returns:
            The prompt to send to the LLM.
        """
        if isinstance(input, str):
            return input

        prompt: str = (
            self._tokenizer.apply_chat_template(  # type: ignore
                input,
                tokenize=False,
                add_generation_prompt=True,
            )
            if input
            else ""
        )
        return super().apply_magpie_pre_query_template(prompt, input)

    @validate_call
    def generate(  # type: ignore
        self,
        inputs: List[Union[StandardInput, str]],
        num_generations: int = 1,
        max_tokens: int = 256,
        logits_processors: Optional[List[Callable]] = None,
        max_kv_size: Optional[int] = None,
        prompt_cache: Optional[Any] = None,
        prefill_step_size: int = 512,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
        temp: float = 0.0,
        top_p: float = 0.0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        top_k: int = -1,
    ) -> List[GenerateOutput]:
        """Generates `num_generations` responses for each input using the text generation
        pipeline.

        Args:
            inputs: the inputs to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            logits_processors: the logits processors to use for the generation. Defaults to
                `None`.
            max_kv_size: the maximum size of the key-value cache. Defaults to `None`.
            prompt_cache: the prompt cache to use for the generation. Defaults to `None`.
            prefill_step_size: the prefill step size. Defaults to `512`.
            kv_bits: the number of bits to use for the key-value cache. Defaults to `None`.
            kv_group_size: the group size for the key-value cache. Defaults to `64`.
            quantized_kv_start: the start of the quantized key-value cache. Defaults to `0`.
            prompt_progress_callback: the callback to use for the generation. Defaults to
                `None`.
            temp: The temperature for text generation. Defaults to `0.0`.
            top_p: The top-p value used for the generation. Defaults to `0.0`.
            min_p: The min-p value used for the generation. Defaults to `0.0`.
            min_tokens_to_keep: Minimum number of tokens to keep for sampling after
                filtering. Must be at least 1. Defaults to `1`.
            top_k: The top-k value used for the generation. Defaults to `-1`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """

        sampler = self._make_sampler(  # type: ignore
            temp=temp,
            top_p=top_p,
            min_p=min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k,
        )
        result = []
        for input in inputs:
            if isinstance(input, tuple):
                input, structured_output = input

            output: List[str] = []
            for _ in range(num_generations):

                configured_processors = list(logits_processors or [])
                if self.structured_output:
                    structured_processors = self._prepare_structured_output(self.structured_output)
                    configured_processors.extend(structured_processors)

                prompt = self.prepare_input(input)
                generation = self._mlx_generate(  # type: ignore
                    prompt=prompt,
                    model=self._model,
                    tokenizer=self._tokenizer,
                    logits_processors=configured_processors,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    max_kv_size=max_kv_size,
                    prompt_cache=prompt_cache,
                    prefill_step_size=prefill_step_size,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                    quantized_kv_start=quantized_kv_start,
                    prompt_progress_callback=prompt_progress_callback,
                )
                output.append(generation)

            result.append(
                prepare_output(
                    generations=output,
                    input_tokens=[compute_tokens(input, self._tokenizer.encode)],  # type: ignore
                    output_tokens=[
                        compute_tokens(
                            text_or_messages=generation,
                            tokenizer=self._tokenizer.encode,  # type: ignore
                        )
                        for generation in output
                    ],
                )
            )
        return result


    def _prepare_structured_output(
            self, structured_output: Optional[OutlinesStructuredOutputType] = None
    ) -> List[Callable]:
        """Creates the appropriate function to filter tokens to generate structured outputs."""
        if structured_output is None:
            return []

        from distilabel.steps.tasks.structured_outputs.outlines import prepare_guided_output
        result = prepare_guided_output(structured_output, "mlx", self._wrapped_model)
        if (schema := result.get("schema")) and self.structured_output:
            self.structured_output["schema"] = schema

        base_processor = result["processor"]

        def mlx_processor(tokens: Any, logits: Any) -> Any:
            # Handle both single and batch inputs uniformly
            is_single = logits.shape[0] == 1
            working_logits = logits[0, :] if is_single else logits[:, -1]

            # Process the logits
            logits_flat = working_logits.reshape(-1)
            processed_logits = base_processor(tokens, logits_flat)

            # Reshape back to original format
            return processed_logits.reshape(1, -1)

        return [mlx_processor]
