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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import Field, FilePath, PrivateAttr, validate_call

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.llms.base import LLM
from distilabel.models.llms.typing import GenerateOutput
from distilabel.models.llms.utils import prepare_output
from distilabel.models.mixins.magpie import MagpieChatTemplateMixin
from distilabel.steps.tasks.typing import FormattedInput, OutlinesStructuredOutputType

if TYPE_CHECKING:
    from llama_cpp import CreateChatCompletionResponse, Llama, LogitsProcessorList

    from distilabel.steps.tasks.typing import (
        FormattedInput,
        StandardInput,
    )


class LlamaCppLLM(LLM, MagpieChatTemplateMixin):
    """llama.cpp LLM implementation running the Python bindings for the C++ code.

    Attributes:
        model_path: contains the path to the GGUF quantized model, compatible with the
            installed version of the `llama.cpp` Python bindings.
        n_gpu_layers: the number of layers to use for the GPU. Defaults to `-1`, meaning that
            the available GPU device will be used.
        chat_format: the chat format to use for the model. Defaults to `None`, which means the
            Llama format will be used.
        n_ctx: the context size to use for the model. Defaults to `512`.
        n_batch: the prompt processing maximum batch size to use for the model. Defaults to `512`.
        seed: random seed to use for the generation. Defaults to `4294967295`.
        verbose: whether to print verbose output. Defaults to `False`.
        structured_output: a dictionary containing the structured output configuration or if more
            fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.
        extra_kwargs: additional dictionary of keyword arguments that will be passed to the
            `Llama` class of `llama_cpp` library. Defaults to `{}`.
        tokenizer_id: the tokenizer Hugging Face Hub repo id or a path to a directory containing
            the tokenizer config files. If not provided, the one associated to the `model`
            will be used. Defaults to `None`.
        use_magpie_template: a flag used to enable/disable applying the Magpie pre-query
            template. Defaults to `False`.
        magpie_pre_query_template: the pre-query template to be applied to the prompt or
            sent to the LLM to generate an instruction or a follow up user message. Valid
            values are "llama3", "qwen2" or another pre-query template provided. Defaults
            to `None`.
        _model: the Llama model instance. This attribute is meant to be used internally and
            should not be accessed directly. It will be set in the `load` method.

    Runtime parameters:
        - `model_path`: the path to the GGUF quantized model.
        - `n_gpu_layers`: the number of layers to use for the GPU. Defaults to `-1`.
        - `chat_format`: the chat format to use for the model. Defaults to `None`.
        - `verbose`: whether to print verbose output. Defaults to `False`.
        - `extra_kwargs`: additional dictionary of keyword arguments that will be passed to the
            `Llama` class of `llama_cpp` library. Defaults to `{}`.

    References:
        - [`llama.cpp`](https://github.com/ggerganov/llama.cpp)
        - [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python)

    Examples:
        Generate text:

        ```python
        from pathlib import Path
        from distilabel.models.llms import LlamaCppLLM

        # You can follow along this example downloading the following model running the following
        # command in the terminal, that will download the model to the `Downloads` folder:
        # curl -L -o ~/Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf

        model_path = "Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf"

        llm = LlamaCppLLM(
            model_path=str(Path.home() / model_path),
            n_gpu_layers=-1,  # To use the GPU if available
            n_ctx=1024,       # Set the context size
        )

        llm.load()

        # Call the model
        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```

        Generate structured data:

        ```python
        from pathlib import Path
        from distilabel.models.llms import LlamaCppLLM

        model_path = "Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf"

        class User(BaseModel):
            name: str
            last_name: str
            id: int

        llm = LlamaCppLLM(
            model_path=str(Path.home() / model_path),  # type: ignore
            n_gpu_layers=-1,
            n_ctx=1024,
            structured_output={"format": "json", "schema": Character},
        )

        llm.load()

        # Call the model
        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
        ```
    """

    model_path: RuntimeParameter[FilePath] = Field(
        default=None, description="The path to the GGUF quantized model.", exclude=True
    )
    n_gpu_layers: RuntimeParameter[int] = Field(
        default=-1,
        description="The number of layers that will be loaded in the GPU.",
    )
    chat_format: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The chat format to use for the model. Defaults to `None`, which means the Llama format will be used.",
    )

    n_ctx: int = 512
    n_batch: int = 512
    seed: int = 4294967295
    verbose: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to print verbose output from llama.cpp library.",
    )
    extra_kwargs: Optional[RuntimeParameter[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Additional dictionary of keyword arguments that will be passed to the"
        " `Llama` class of `llama_cpp` library. See all the supported arguments at: "
        "https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__",
    )
    structured_output: Optional[RuntimeParameter[OutlinesStructuredOutputType]] = Field(
        default=None,
        description="The structured output format to use across all the generations.",
    )
    tokenizer_id: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The tokenizer Hugging Face Hub repo id or a path to a directory containing"
        " the tokenizer config files. If not provided, the one associated to the `model`"
        " will be used.",
    )
    use_magpie_template: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to use the Magpie pre-query template or not.",
    )
    magpie_pre_query_template: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The pre-query template to use for the model. Valid values are "
        "`llama3`, `qwen2` or another pre-query template provided.",
    )
    _logits_processor: Optional["LogitsProcessorList"] = PrivateAttr(default=None)
    _model: Optional["Llama"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the `Llama` model from the `model_path`."""
        try:
            from llama_cpp import Llama
        except ImportError as ie:
            raise ImportError(
                "The `llama_cpp` package is required to use the `LlamaCppLLM` class."
            ) from ie

        self._model = Llama(
            model_path=self.model_path.as_posix(),  # type: ignore
            seed=self.seed,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            chat_format=self.chat_format,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
            **self.extra_kwargs,
        )

        if self.structured_output:
            self._logits_processor = self._prepare_structured_output(
                self.structured_output
            )

        if self.use_magpie_template or self.magpie_pre_query_template:
            if not self.tokenizer_id:
                raise ValueError(
                    "The Hugging Face Hub repo id or a path to a directory containing"
                    " the tokenizer config files is required when using the `use_magpie_template`"
                    " or `magpie_pre_query_template` runtime parameters."
                )

        if self.tokenizer_id:
            try:
                from transformers import AutoTokenizer
            except ImportError as ie:
                raise ImportError(
                    "Transformers is not installed. Please install it using `pip install transformers`."
                ) from ie
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

        # NOTE: Here because of the custom `logging` interface used, since it will create the logging name
        # out of the model name, which won't be available until the `Llama` instance is created.
        super().load()

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self._model.model_path  # type: ignore

    def _generate_chat_completion(
        self,
        input: FormattedInput,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        extra_generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "CreateChatCompletionResponse":
        return self._model.create_chat_completion(  # type: ignore
            messages=input,  # type: ignore
            max_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            logits_processor=self._logits_processor,
            **(extra_generation_kwargs or {}),
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

    def _generate_with_text_generation(
        self,
        input: FormattedInput,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        extra_generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "CreateChatCompletionResponse":
        prompt = self.prepare_input(input)
        return self._model.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            logits_processor=self._logits_processor,
            **(extra_generation_kwargs or {}),
        )

    @validate_call
    def generate(  # type: ignore
        self,
        inputs: List[FormattedInput],
        num_generations: int = 1,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        extra_generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[GenerateOutput]:
        """Generates `num_generations` responses for the given input using the Llama model.

        Args:
            inputs: a list of inputs in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            frequency_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            extra_generation_kwargs: dictionary with additional arguments to be passed to
                the `create_chat_completion` method. Reference at
                https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        structured_output = None
        batch_outputs = []
        for input in inputs:
            if isinstance(input, tuple):
                input, structured_output = input
            elif self.structured_output:
                structured_output = self.structured_output

            outputs = []
            output_tokens = []
            for _ in range(num_generations):
                # NOTE(plaguss): There seems to be a bug in how the logits processor
                # is used. Basically it consumes the FSM internally, and it isn't reinitialized
                # after each generation, so subsequent calls yield nothing. This is a workaround
                # until is fixed in the `llama_cpp` or `outlines` libraries.
                if structured_output:
                    self._logits_processor = self._prepare_structured_output(
                        structured_output
                    )
                if self.tokenizer_id is None:
                    completion = self._generate_chat_completion(
                        input,
                        max_new_tokens,
                        frequency_penalty,
                        presence_penalty,
                        temperature,
                        top_p,
                        extra_generation_kwargs,
                    )
                    outputs.append(completion["choices"][0]["message"]["content"])
                    output_tokens.append(completion["usage"]["completion_tokens"])
                else:
                    completion: "CreateChatCompletionResponse" = (
                        self._generate_with_text_generation(  # type: ignore
                            input,
                            max_new_tokens,
                            frequency_penalty,
                            presence_penalty,
                            temperature,
                            top_p,
                            extra_generation_kwargs,
                        )
                    )
                    outputs.append(completion["choices"][0]["text"])
                    output_tokens.append(completion["usage"]["completion_tokens"])
            batch_outputs.append(
                prepare_output(
                    outputs,
                    input_tokens=[completion["usage"]["prompt_tokens"]]
                    * num_generations,
                    output_tokens=output_tokens,
                )
            )

        return batch_outputs

    def _prepare_structured_output(
        self, structured_output: Optional[OutlinesStructuredOutputType] = None
    ) -> Union["LogitsProcessorList", None]:
        """Creates the appropriate function to filter tokens to generate structured outputs.

        Args:
            structured_output: the configuration dict to prepare the structured output.

        Returns:
            The callable that will be used to guide the generation of the model.
        """
        from distilabel.steps.tasks.structured_outputs.outlines import (
            prepare_guided_output,
        )

        result = prepare_guided_output(structured_output, "llamacpp", self._model)
        if (schema := result.get("schema")) and self.structured_output:
            self.structured_output["schema"] = schema
        return result["processor"]
