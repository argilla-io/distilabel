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

from typing import TYPE_CHECKING, List, Optional

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
        chat_format: the chat format to use for the model. Defaults to `chatml`.
        model_path: contains the path to the GGUF quantized model, compatible with the
            installed version of the `llama.cpp` Python bindings.
        n_gpu_layers: the number of layers to use for the GPU. Defaults to `-1`, meaning that
            the available GPU device will be used.
        verbose: whether to print verbose output. Defaults to `False`.
        _model: the Llama model instance. This attribute is meant to be used internally and
            should not be accessed directly. It will be set in the `load` method.

    Runtime parameters:
        - `model_path`: the path to the GGUF quantized model.
        - `n_gpu_layers`: the number of layers to use for the GPU. Defaults to `-1`.
        - `verbose`: whether to print verbose output. Defaults to `False`.
    """

    chat_format: str = "chatml"
    model_path: RuntimeParameter[FilePath] = Field(
        default=None, description="The path to the GGUF quantized model."
    )
    n_gpu_layers: RuntimeParameter[int] = Field(
        default=-1,
        description="The number of layers that will be loaded in the GPU.",
    )
    verbose: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to print verbose output from llama.cpp library.",
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
            model_path=self.model_path.as_posix(),
            chat_format=self.chat_format,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )

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
                    )
                )
                outputs.append(chat_completions["choices"][0]["message"]["content"])
            batch_outputs.append(outputs)
        return batch_outputs
