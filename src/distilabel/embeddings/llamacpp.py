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

from pydantic import PrivateAttr

from distilabel.embeddings.base import Embeddings
from distilabel.llms.mixins.cuda_device_placement import CudaDevicePlacementMixin

if TYPE_CHECKING:
    from llama_cpp import Llama as _LlamaCpp


class LlamaCppEmbeddings(Embeddings, CudaDevicePlacementMixin):
    """`LlamaCpp` library implementation for embedding generation.

    Attributes:
        model: the model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        hub_repository_id: the Hugging Face Hub repository id.
        _model: the `Llama` model instance. This attribute is meant to be used internally
            and should not be accessed directly. It will be set in the `load` method.

    References:
        - [Offline inference embeddings](https://llama-cpp-python.readthedocs.io/en/stable/#embeddings)

    Examples:
        Generating sentence embeddings:

        ```python
        from distilabel.embeddings import LlamaCppEmbeddings

        embeddings = LlamaCppEmbeddings(model="second-state/all-MiniLM-L6-v2-Q2_K.gguf")

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        # [
        #   [-0.05447685346007347, -0.01623094454407692, ...],
        #   [4.4889533455716446e-05, 0.044016145169734955, ...],
        # ]
        ```
    """

    model: str
    hub_repository_id: Union[None, str] = None
    disable_cuda_device_placement: bool = True
    model_kwargs: Optional[Dict[str, Any]] = None
    verbose: bool = False
    _model: Union["_LlamaCpp", None] = PrivateAttr(None)

    def load(self) -> None:
        """Loads the `gguf` model using either the path or the Hugging Face Hub repository id."""
        super().load()

        CudaDevicePlacementMixin.load(self)

        try:
            from llama_cpp import Llama as _LlamaCpp
        except ImportError as ie:
            raise ImportError(
                "`llama-cpp-python` package is not installed. Please install it using"
                " `pip install llama-cpp-python`."
            ) from ie

        if self.hub_repository_id is not None:
            self._model = _LlamaCpp.from_pretrained(
                repo_id=self.hub_repository_id,
                filename=self.model,
                verbose=self.verbose,
                embedding=True,
            )
        else:
            try:
                self._logger.info(f"Attempting to load model from: {self.model_name}")
                self._model = _LlamaCpp(
                    model_path=self.model_name,
                    verbose=self.verbose,
                    embedding=True,
                    kwargs=self.model_kwargs,
                )
                self._logger.info(f"self._model: {self._model}")
                self._logger.info("Model loaded successfully")
            except Exception as e:
                self._logger.error(f"Failed to load model: {str(e)}")
                raise

    def unload(self) -> None:
        """Unloads the `gguf` model."""
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
        if self._model is None:
            self._logger.error("Model is not initialized")
            raise ValueError(
                "Model is not initialized. Please check the initialization process."
            )

        try:
            return self._model.create_embedding(inputs)["data"]
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            raise
