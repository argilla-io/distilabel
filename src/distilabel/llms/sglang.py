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
    Dict,
    List,
    Optional,
)

from pydantic import Field, PrivateAttr, validate_call

from distilabel.llms.base import LLM
from distilabel.llms.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.llms.mixins.magpie import MagpieChatTemplateMixin
from distilabel.llms.typing import GenerateOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import FormattedInput, OutlinesStructuredOutputType

if TYPE_CHECKING:
    from openai import OpenAI  # noqa

    from distilabel.steps.tasks.typing import StandardInput


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
    revision: Optional[str] = None

    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    tokenizer_revision: Optional[str] = None
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

    _model: Any = PrivateAttr(None)
    _tokenizer: Any = PrivateAttr(None)

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
                'SGLang is not installed. Please install it using `pip install "sglang[all]"`.'
                " Also, install FlashInfer CUDA kernels using:\n"
                "`pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/`"
            ) from ie

        self._model = Runtime(
            model_path=self.model,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization=self.quantization,
            revision=self.revision,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            tokenizer_revision=self.tokenizer_revision,
            skip_tokenizer_init=self.skip_tokenizer_init,
            load_format=self.load_format,
            kv_cache_dtype=self.kv_cache_dtype,
            context_length=self.context_length,
            served_model_name=self.served_model_name,
            is_embedding=self.is_embedding,
            seed=self.seed,
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

    @validate_call
    def generate(
        self,
        inputs: List[FormattedInput],
        num_generations: int = 1,
        max_new_tokens: int = 128,
        # Add other relevant parameters here
    ) -> List[GenerateOutput]:
        """Generates responses for each input."""
        # Implement generation logic here
        pass


# You can add a ClientSGLang class here if needed, similar to ClientvLLM
