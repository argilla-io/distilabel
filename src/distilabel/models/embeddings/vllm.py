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

from pydantic import Field, PrivateAttr

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.embeddings.base import Embeddings
from distilabel.models.mixins.cuda_device_placement import CudaDevicePlacementMixin

if TYPE_CHECKING:
    from vllm import LLM as _vLLM


class vLLMEmbeddings(Embeddings, CudaDevicePlacementMixin):
    """`vllm` library implementation for embedding generation.

    Attributes:
        model: the model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        dtype: the data type to use for the model. Defaults to `auto`.
        trust_remote_code: whether to trust the remote code when loading the model. Defaults
            to `False`.
        quantization: the quantization mode to use for the model. Defaults to `None`.
        revision: the revision of the model to load. Defaults to `None`.
        enforce_eager: whether to enforce eager execution. Defaults to `True`.
        seed: the seed to use for the random number generator. Defaults to `0`.
        extra_kwargs: additional dictionary of keyword arguments that will be passed to the
            `LLM` class of `vllm` library. Defaults to `{}`.
        _model: the `vLLM` model instance. This attribute is meant to be used internally
            and should not be accessed directly. It will be set in the `load` method.

    References:
        - [Offline inference embeddings](https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference_embedding.html)

    Examples:
        Generating sentence embeddings:

        ```python
        from distilabel.models import vLLMEmbeddings

        embeddings = vLLMEmbeddings(model="intfloat/e5-mistral-7b-instruct")

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        # [
        #   [-0.05447685346007347, -0.01623094454407692, ...],
        #   [4.4889533455716446e-05, 0.044016145169734955, ...],
        # ]
        ```
    """

    model: str
    dtype: str = "auto"
    trust_remote_code: bool = False
    quantization: Optional[str] = None
    revision: Optional[str] = None

    enforce_eager: bool = True

    seed: int = 0

    extra_kwargs: Optional[RuntimeParameter[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Additional dictionary of keyword arguments that will be passed to the"
        " `vLLM` class of `vllm` library. See all the supported arguments at: "
        "https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py",
    )

    _model: "_vLLM" = PrivateAttr(None)

    def load(self) -> None:
        """Loads the `vLLM` model using either the path or the Hugging Face Hub repository id."""
        super().load()

        CudaDevicePlacementMixin.load(self)

        try:
            from vllm import LLM as _vLLM

        except ImportError as ie:
            raise ImportError(
                "vLLM is not installed. Please install it using `pip install 'distilabel[vllm]'`."
            ) from ie

        self._model = _vLLM(
            self.model,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization=self.quantization,
            revision=self.revision,
            enforce_eager=self.enforce_eager,
            seed=self.seed,
            **self.extra_kwargs,  # type: ignore
        )

    def unload(self) -> None:
        """Unloads the `vLLM` model."""
        CudaDevicePlacementMixin.unload(self)
        super().unload()

    @property
    def model_name(self) -> str:
        """Returns the name of the model."""
        return self.model

    def encode(self, inputs: List[str]) -> List[List[Union[int, float]]]:
        """Generates embeddings for the provided inputs.

        Args:
            inputs: a list of texts for which an embedding has to be generated.

        Returns:
            The generated embeddings.
        """
        return [output.outputs.embedding for output in self._model.encode(inputs)]
