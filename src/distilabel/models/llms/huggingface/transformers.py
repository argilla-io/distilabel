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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from pydantic import Field, PrivateAttr, SecretStr, validate_call

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.llms.base import LLM
from distilabel.models.llms.typing import GenerateOutput
from distilabel.models.llms.utils import compute_tokens, prepare_output
from distilabel.models.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.models.mixins.magpie import MagpieChatTemplateMixin
from distilabel.steps.tasks.structured_outputs.outlines import outlines_below_0_1_0
from distilabel.steps.tasks.typing import OutlinesStructuredOutputType, StandardInput
from distilabel.utils.huggingface import HF_TOKEN_ENV_VAR

if TYPE_CHECKING:
    from transformers import Pipeline
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils import PreTrainedTokenizer

    from distilabel.models.llms.typing import HiddenState


class TransformersLLM(LLM, MagpieChatTemplateMixin, CudaDevicePlacementMixin):
    """Hugging Face `transformers` library LLM implementation using the text generation
    pipeline.

    Attributes:
        model: the model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        revision: if `model` refers to a Hugging Face Hub repository, then the revision
            (e.g. a branch name or a commit id) to use. Defaults to `"main"`.
        torch_dtype: the torch dtype to use for the model e.g. "float16", "float32", etc.
            Defaults to `"auto"`.
        trust_remote_code: whether to allow fetching and executing remote code fetched
            from the repository in the Hub. Defaults to `False`.
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
        structured_output: a dictionary containing the structured output configuration or if more
            fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.
        use_magpie_template: a flag used to enable/disable applying the Magpie pre-query
            template. Defaults to `False`.
        magpie_pre_query_template: the pre-query template to be applied to the prompt or
            sent to the LLM to generate an instruction or a follow up user message. Valid
            values are "llama3", "qwen2" or another pre-query template provided. Defaults
            to `None`.

    Icon:
        `:hugging:`

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import TransformersLLM

        llm = TransformersLLM(model="microsoft/Phi-3-mini-4k-instruct")

        llm.load()

        # Call the model
        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```
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
    token: Optional[SecretStr] = Field(
        default_factory=lambda: os.getenv(HF_TOKEN_ENV_VAR)
    )
    structured_output: Optional[RuntimeParameter[OutlinesStructuredOutputType]] = Field(
        default=None,
        description="The structured output format to use across all the generations.",
    )

    _pipeline: Optional["Pipeline"] = PrivateAttr(...)
    _prefix_allowed_tokens_fn: Union[Callable, None] = PrivateAttr(default=None)
    _logits_processor: Union[Callable, None] = PrivateAttr(default=None)

    def load(self) -> None:
        """Loads the model and tokenizer and creates the text generation pipeline. In addition,
        it will configure the tokenizer chat template."""
        if self.device == "cuda":
            CudaDevicePlacementMixin.load(self)

        try:
            from transformers import pipeline
        except ImportError as ie:
            raise ImportError(
                "Transformers is not installed. Please install it using `pip install transformers`."
            ) from ie

        token = self.token.get_secret_value() if self.token is not None else self.token

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
            token=token,
            return_full_text=False,
        )

        if self.chat_template is not None:
            self._pipeline.tokenizer.chat_template = self.chat_template  # type: ignore

        if self._pipeline.tokenizer.pad_token is None:  # type: ignore
            self._pipeline.tokenizer.pad_token = self._pipeline.tokenizer.eos_token  # type: ignore

        if self.structured_output:
            processor = self._prepare_structured_output(self.structured_output)
            if outlines_below_0_1_0:
                self._prefix_allowed_tokens_fn = processor
            else:
                self._logits_processor = [processor]

        super().load()

    def unload(self) -> None:
        """Unloads the `vLLM` model."""
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
        if self._pipeline.tokenizer.chat_template is None:  # type: ignore
            return input[0]["content"]

        prompt: str = (
            self._pipeline.tokenizer.apply_chat_template(  # type: ignore
                input,  # type: ignore
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
        inputs: List[StandardInput],
        num_generations: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        repetition_penalty: float = 1.1,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
    ) -> List[GenerateOutput]:
        """Generates `num_generations` responses for each input using the text generation
        pipeline.

        Args:
            inputs: a list of inputs in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximum number of new tokens that the model will generate.
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
        prepared_inputs = [self.prepare_input(input=input) for input in inputs]

        outputs: List[List[Dict[str, str]]] = self._pipeline(  # type: ignore
            prepared_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_generations,
            prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
            logits_processor=self._logits_processor,
        )
        llm_output = [
            [generation["generated_text"] for generation in output]
            for output in outputs
        ]

        result = []
        for input, output in zip(inputs, llm_output):
            result.append(
                prepare_output(
                    output,
                    input_tokens=[
                        compute_tokens(input, self._pipeline.tokenizer.encode)
                    ],
                    output_tokens=[
                        compute_tokens(row, self._pipeline.tokenizer.encode)
                        for row in output
                    ],
                )
            )

        return result

    def get_last_hidden_states(
        self, inputs: List["StandardInput"]
    ) -> List["HiddenState"]:
        """Gets the last `hidden_states` of the model for the given inputs. It doesn't
        execute the task head.

        Args:
            inputs: a list of inputs in chat format to generate the embeddings for.

        Returns:
            A list containing the last hidden state for each sequence using a NumPy array
            with shape [num_tokens, hidden_size].
        """
        model: "PreTrainedModel" = (
            self._pipeline.model.model  # type: ignore
            if hasattr(self._pipeline.model, "model")  # type: ignore
            else next(self._pipeline.model.children())  # type: ignore
        )
        tokenizer: "PreTrainedTokenizer" = self._pipeline.tokenizer  # type: ignore
        input_ids = tokenizer(
            [self.prepare_input(input) for input in inputs],  # type: ignore
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        last_hidden_states = model(**input_ids)["last_hidden_state"]

        return [
            seq_last_hidden_state[attention_mask.bool(), :].detach().cpu().numpy()
            for seq_last_hidden_state, attention_mask in zip(
                last_hidden_states,
                input_ids["attention_mask"],  # type: ignore
            )
        ]

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

        result = prepare_guided_output(
            structured_output, "transformers", self._pipeline
        )
        if schema := result.get("schema"):
            self.structured_output["schema"] = schema
        return result["processor"]
