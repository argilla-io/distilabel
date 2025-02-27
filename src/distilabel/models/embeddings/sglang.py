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
    from sglang import Engine as _SGLang


class SGLangEmbeddings(Embeddings, CudaDevicePlacementMixin):
    """`sglang` library implementation for embedding generation.

    Attributes:
        model: the model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        dtype: the data type to use for the model. Defaults to `auto`.
        trust_remote_code: whether to trust the remote code when loading the model. Defaults
            to `False`.
        quantization: the quantization mode to use for the model. Defaults to `None`.
        revision: the revision of the model to load. Defaults to `None`.
        seed: the seed to use for the random number generator. Defaults to `0`.
        extra_kwargs: additional dictionary of keyword arguments that will be passed to the
            `Engine` class of `sglang` library. Defaults to `{}`.
        _model: the `SGLang` model instance. This attribute is meant to be used internally
            and should not be accessed directly. It will be set in the `load` method.

    References:
        - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py

    Examples:
        Generating sentence embeddings:

        ```python
        if __name__ == "__main__":

            from distilabel.models import SGLangEmbeddings
            embeddings = SGLangEmbeddings(model="intfloat/e5-mistral-7b-instruct")
            embeddings.load()
            results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
            print(results)
            # [
            #   [0.0203704833984375, -0.0060882568359375, ...],
            #   [0.02398681640625, 0.0177001953125 ...],
            # ]
        ```
    """

    model: str
    dtype: str = "auto"
    trust_remote_code: bool = False
    quantization: Optional[str] = None
    revision: Optional[str] = None

    seed: int = 0

    extra_kwargs: Optional[RuntimeParameter[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Additional dictionary of keyword arguments that will be passed to the"
        " `Engine` class of `sglang` library. See all the supported arguments at: "
        "https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py",
    )

    _model: "_SGLang" = PrivateAttr(None)

    def load(self) -> None:
        """Loads the `sglang` model using either the path or the Hugging Face Hub repository id."""
        super().load()

        CudaDevicePlacementMixin.load(self)

        try:
            from sglang import Engine as _SGLang
        except ImportError as err:
            raise ImportError(
                "sglang is not installed. Please install it with sglang document https://docs.sglang.ai/start/install.html."
            ) from err

        self._model = _SGLang(
            model_path=self.model,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization=self.quantization,
            revision=self.revision,
            random_seed=self.seed,
            **self.extra_kwargs,  # type: ignore
        )

    def unload(self) -> None:
        """Unloads the `SGLang` model."""
        self._cleanup_sglang_model()
        self._model = None  # type: ignore
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
        return [output["embedding"] for output in self._model.encode(inputs)]
