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

from pydantic import Field, PrivateAttr, validate_call

from distilabel.llms.base import LLM
from distilabel.llms.chat_templates import CHATML_TEMPLATE
from distilabel.llms.mixins import CudaDevicePlacementMixin
from distilabel.llms.typing import GenerateOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import ChatType

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from vllm import LLM as _vLLM


SamplingParams = None


class vLLM(LLM, CudaDevicePlacementMixin):
    """`vLLM` library LLM implementation.

    Attributes:
        model: the model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        model_kwargs: additional dictionary of keyword arguments that will be passed to
            the `LLM` class of `vllm` library.
        chat_template: a chat template that will be used to build the prompts before
            sending them to the model. If not provided, the chat template defined in the
            tokenizer config will be used. If not provided and the tokenizer doesn't have
            a chat template, then ChatML template will be used. Defaults to `None`.
        _model: the `vLLM` model instance. This attribute is meant to be used internally
            and should not be accessed directly. It will be set in the `load` method.
        _tokenizer: the tokenizer instance used to format the prompt before passing it to
            the `LLM`. This attribute is meant to be used internally and should not be
            accessed directly. It will be set in the `load` method.

    Runtime parameters:
        - `model_kwargs`: additional dictionary of keyword arguments that will be passed to
            the `LLM` class of `vllm` library.
    """

    model: str
    model_kwargs: Optional[RuntimeParameter[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Additional dictionary of keyword arguments that will be passed to the"
        " `LLM` class of `vllm` library.",
    )
    chat_template: Optional[str] = None

    _model: Optional["_vLLM"] = PrivateAttr(...)
    _tokenizer: Optional["PreTrainedTokenizer"] = PrivateAttr(...)

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
            from vllm import SamplingParams as _SamplingParams

            global SamplingParams
            SamplingParams = _SamplingParams
        except ImportError as ie:
            raise ImportError(
                "vLLM is not installed. Please install it using `pip install vllm`."
            ) from ie

        self._model = _vLLM(self.model, **self.model_kwargs)  # type: ignore
        self._tokenizer = self._model.get_tokenizer()  # type: ignore

        if self.chat_template is not None:
            self._tokenizer.chat_template = self.chat_template  # type: ignore
        elif (
            self._tokenizer.chat_template is None  # type: ignore
            and self._tokenizer.default_chat_template is None  # type: ignore
        ):
            self._tokenizer.chat_template = CHATML_TEMPLATE

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    def prepare_input(self, input: "ChatType") -> str:
        """Prepares the input by applying the chat template to the input, which is formatted
        as an OpenAI conversation, and adding the generation prompt.
        """
        return self._tokenizer.apply_chat_template(  # type: ignore
            input,  # type: ignore
            tokenize=False,
            add_generation_prompt=True,  # type: ignore
        )

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
        top_k: int = -1,
        extra_sampling_params: Optional[Dict[str, Any]] = None,
    ) -> List[GenerateOutput]:
        """Generates `num_generations` responses for each input using the text generation
        pipeline.

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
            top_k: the top-k value to use for the generation. Defaults to `0`.
            extra_sampling_params: dictionary with additional arguments to be passed to
                the `SamplingParams` class from `vllm`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        if extra_sampling_params is None:
            extra_sampling_params = {}

        sampling_params = SamplingParams(  # type: ignore
            n=num_generations,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            **extra_sampling_params,
        )

        prepared_inputs = [self.prepare_input(input) for input in inputs]
        batch_outputs = self._model.generate(  # type: ignore
            prepared_inputs,
            sampling_params,
            use_tqdm=False,  # type: ignore
        )
        return [
            [output.text for output in outputs.outputs] for outputs in batch_outputs
        ]
