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
)

from pydantic import (
    PrivateAttr,
    validate_call,
)

from distilabel.models.llms.base import LLM
from distilabel.models.llms.typing import GenerateOutput
from distilabel.models.llms.utils import compute_tokens, prepare_output
from distilabel.models.mixins.magpie import MagpieChatTemplateMixin
from distilabel.steps.tasks.typing import (
    StandardInput,
)

if TYPE_CHECKING:
    import mlx.nn as nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper


class MlxLLM(LLM, MagpieChatTemplateMixin):
    """Apple MLX LLM implementation.

    Attributes:
        path_or_hf_repo: the path to the model or the Hugging Face Hub repo id.
        tokenizer_config: the tokenizer configuration.
        model_config: the model configuration.
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

        llm = MlxLLM(model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

        llm.load()

        # Call the model
        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```
    """

    path_or_hf_repo: str
    tokenizer_config: Dict[str, Any] = {}
    model_config: Dict[str, Any] = {}
    adapter_path: Optional[str] = None

    _mlx_generate: Optional[Callable] = PrivateAttr(default=None)
    _model: Optional["nn.Module"] = PrivateAttr(...)
    _tokenizer: Optional["TokenizerWrapper"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the model and tokenizer and creates the text generation pipeline. In addition,
        it will configure the tokenizer chat template."""
        try:
            import mlx  # noqa
            from mlx_lm import generate, load
        except ImportError as ie:
            raise ImportError(
                "MLX is not installed. Please install it using `pip install 'distilabel[mlx]'`."
            ) from ie

        self._model, self._tokenizer = load(
            self.path_or_hf_repo,
            tokenizer_config=self.tokenizer_config,
            model_config=self.model_config,
            adapter_path=self.adapter_path,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._mlx_generate = generate

        super().load()

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.path_or_hf_repo

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
                input,
                tokenize=False,
                add_generation_prompt=True,
            )
            if input
            else ""
        )
        return super().apply_magpie_pre_query_template(prompt, input)

    @validate_call
    def generate(
        self,
        inputs: List[StandardInput],
        num_generations: int = 1,
        max_tokens: int = 256,
        sampler: Optional[Callable] = None,
        logits_processors: Optional[List[Callable]] = None,
        max_kv_size: Optional[int] = None,
        prompt_cache: Optional[Any] = None,
        prefill_step_size: int = 512,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
        temp: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        min_tokens_to_keep: Optional[int] = None,
    ) -> List[GenerateOutput]:
        """Generates `num_generations` responses for each input using the text generation
        pipeline.

        Args:
            inputs: the inputs to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            sampler: the sampler to use for the generation. Defaults to `None`.
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
            temp: the temperature to use for the generation. Defaults to `None`.
            repetition_penalty: the repetition penalty to use for the generation. Defaults to
                `None`.
            repetition_context_size: the context size for the repetition penalty. Defaults to
                `None`.
            top_p: the top-p value to use for the generation. Defaults to `None`.
            min_p: the minimum p value to use for the generation. Defaults to `None`.
            min_tokens_to_keep: the minimum number of tokens to keep. Defaults to `None`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        structured_output = None
        result = []
        for input in inputs:
            if isinstance(input, tuple):
                input, structured_output = input

            output: List[str] = []
            for _ in range(num_generations):
                if structured_output:  # will raise a NotImplementedError
                    self._prepare_structured_output(structured_output)
                prompt = self.prepare_input(input)
                generation = self._mlx_generate(
                    prompt=prompt,
                    model=self._model,
                    tokenizer=self._tokenizer,
                    logits_processors=logits_processors,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    max_kv_size=max_kv_size,
                    prompt_cache=prompt_cache,
                    prefill_step_size=prefill_step_size,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                    quantized_kv_start=quantized_kv_start,
                    prompt_progress_callback=prompt_progress_callback,
                    temp=temp,
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=repetition_context_size,
                    top_p=top_p,
                    min_p=min_p,
                    min_tokens_to_keep=min_tokens_to_keep,
                )

                output.append(generation)

            result.append(
                prepare_output(
                    output,
                    input_tokens=[compute_tokens(input, self._tokenizer.encode)],
                    output_tokens=[
                        compute_tokens(
                            text_or_messages=generation,
                            tokenizer=self._tokenizer.encode,
                        )
                        for generation in output
                    ],
                )
            )
        return result
