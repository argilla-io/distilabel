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

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import Field, FilePath, PrivateAttr, validate_call

from distilabel.llms.base import LLM
from distilabel.llms.typing import GenerateOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import ChatType

if TYPE_CHECKING:
    from llama_cpp import CreateChatCompletionResponse, Llama


class LlamaCppLLM(LLM):
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
        extra_kwargs: additional dictionary of keyword arguments that will be passed to the
            `Llama` class of `llama_cpp` library. Defaults to `{}`.
        _model: the Llama model instance. This attribute is meant to be used internally and
            should not be accessed directly. It will be set in the `load` method.

    Runtime parameters:
        - `model_path`: the path to the GGUF quantized model.
        - `n_gpu_layers`: the number of layers to use for the GPU. Defaults to `-1`.
        - `chat_format`: the chat format to use for the model. Defaults to `None`.
        - `verbose`: whether to print verbose output. Defaults to `False`.
        - `extra_kwargs`: additional dictionary of keyword arguments that will be passed to the
            `Llama` class of `llama_cpp` library. Defaults to `{}`.
    """

    model_path: RuntimeParameter[FilePath] = Field(
        default=None, description="The path to the GGUF quantized model."
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
        " `Llama` class of `llama_cpp` library. See all the suported arguments at: "
        "https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__",
    )

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

        # NOTE: Here because of the custom `logging` interface used, since it will create the logging name
        # out of the model name, which won't be available until the `Llama` instance is created.
        super().load()

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self._model.model_path  # type: ignore

    @validate_call
    def generate(  # type: ignore
        self,
        inputs: List[ChatType],
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
        batch_outputs = []
        for input in inputs:
            outputs = []
            for _ in range(num_generations):
                chat_completions: "CreateChatCompletionResponse" = (
                    self._model.create_chat_completion(  # type: ignore
                        messages=input,  # type: ignore
                        max_tokens=max_new_tokens,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        temperature=temperature,
                        top_p=top_p,
                        **(extra_generation_kwargs or {}),
                    )
                )
                outputs.append(chat_completions["choices"][0]["message"]["content"])
            batch_outputs.append(outputs)
        return batch_outputs
