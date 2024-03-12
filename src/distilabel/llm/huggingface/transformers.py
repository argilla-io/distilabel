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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import PrivateAttr
from transformers import Pipeline, pipeline

from distilabel.llm.base import LLM

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils import PreTrainedTokenizer

    from distilabel.llm.typing import HiddenStatesVector
    from distilabel.steps.task.typing import ChatType


class TransformersLLM(LLM):
    """Hugging Face `transformers` library LLM implementation using the text generation
    pipeline.

    Attributes:
        model: the model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        revision: if `model` refers to a Hugging Face Hub repository, then the revision
            (e.g. a branch name or a commit id) to use. Defaults to `"main"`.
        torch_dtype: the torch dtype to use for the model e.g. "float16", "float32", etc.
            Defaults to `"auto"`.
        trust_remote_code: whether to trust or not remote (code in the Hugging Face Hub
            repository) code to load the model. Defaults to `False`.
        model_kwargs: additional dictionary of keyword arguments that will be passed to
            the `from_pretrained` method of the model.
        tokenizer: the tokenizer Hugging Face Hub repo id or a path to a directory containing
            the tokenizer config files. If not provided, the one associated to the `model`
            will be used. Defaults to `None`.
        use_fast: whether to use a fast tokenizer or not. Defaults to `True`.
        chat_template: a chat template that will be used to build the prompts before
            sending them to the model. If not provided, the chat template defined in the
            tokenizer config will be used. If not provided and the tokenizer doesn't have
            a chat template, then ChatML template will be used. Defaults to `None`.
        device: the name or index of the device where the model will be loaded. Defaults
            to `None`.
        device_map: a dictionary mapping each layer of the model to a device, or a mode
            like `"sequential"` or `"auto"`. Defaults to `None`.
        token: the Hugging Face Hub token that will be used to authenticate to the Hugging
            Face Hub. If not provided, the `HF_TOKEN` environment or `huggingface_hub` package
            local configuration will be used. Defaults to `None`.
    """

    model: str
    revision: str = "main"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer: Optional[str] = None
    use_fast: bool = True
    chat_template: Optional[str] = None
    device: Optional[Union[str, int]] = None
    device_map: Optional[Union[str, Dict[str, Any]]] = None
    token: Optional[str] = None

    _pipeline: Optional["Pipeline"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the model and tokenizer and creates the text generation pipeline. In addition,
        it will configure the tokenizer chat template."""

        self._pipeline = pipeline(
            "text-generation",
            model=self.model,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            model_kwargs=self.model_kwargs or {},
            tokenizer=self.tokenizer or self.model,
            use_fast=self.use_fast,
            device=self.device,
            device_map=self.device_map,
            token=self.token or os.getenv("HF_TOKEN"),
            return_full_text=False,
        )

        if self.chat_template is not None:
            self._pipeline.tokenizer.chat_template = self.chat_template  # type: ignore
        elif (
            self._pipeline.tokenizer.chat_template is None  # type: ignore
            and self._pipeline.tokenizer.default_chat_template is None  # type: ignore
        ):
            self._pipeline.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # type: ignore

    @property
    def model_name(self) -> str:
        return self.model

    def prepare_input(self, input: "ChatType") -> str:
        return self._pipeline.tokenizer.apply_chat_template(  # type: ignore
            input,  # type: ignore
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(  # type: ignore
        self,
        inputs: List["ChatType"],
        num_generations: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        repetition_penalty: float = 1.1,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
    ) -> List[List[str]]:
        """Generates `num_generations` responses for each input using the text generation
        pipeline.

        Args:
            inputs: a list of inputs in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximun number of new tokens that the model will generate.
                Defaults to `128`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            repetition_penalty: the repetition penalty to use for the generation. Defaults
                to `1.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            top_k: the top-k value to use for the generation. Defaults to `0`.
            do_sample: whether to use sampling or not. Defaults to `True`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        outputs: List[List[Dict[str, str]]] = self._pipeline(  # type: ignore
            [self.prepare_input(input=input) for input in inputs],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_generations,
        )
        return [
            [generation["generated_text"] for generation in output]
            for output in outputs
        ]

    def get_last_hidden_states(self, inputs: List["ChatType"]) -> "HiddenStatesVector":
        """Gets the last `hidden_states` of the model for the given inputs. It doesn't
        execute the task head.

        Returns:
            A numpy array of shape `(len(inputs), num_tokens, hidden_size)` containing the
            last hidden states of the model for each input.
        """
        # `model.model` so the `LMHead` is not used
        model: "PreTrainedModel" = self._pipeline.model.model  # type: ignore
        tokenizer: "PreTrainedTokenizer" = self._pipeline.tokenizer  # type: ignore
        formatted_inputs = [self.prepare_input(input=input) for input in inputs]
        last_hidden_states = model(
            **tokenizer(formatted_inputs, return_tensors="pt", padding=True).to(
                model.device
            )
        )["last_hidden_state"]
        return last_hidden_states.cpu().detach().numpy()
