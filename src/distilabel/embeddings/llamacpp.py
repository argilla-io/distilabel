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

from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import Field, PrivateAttr

from distilabel.embeddings.base import Embeddings
from distilabel.llms.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.mixins.runtime_parameters import RuntimeParameter

if TYPE_CHECKING:
    from llama_cpp import Llama as _LlamaCpp


class LlamaCppEmbeddings(Embeddings, CudaDevicePlacementMixin):
    """`LlamaCpp` library implementation for embedding generation.

    Attributes:
        model: contains the path to the GGUF quantized model, compatible with the
            installed version of the `llama.cpp` Python bindings.
        hub_repository_id: the Hugging Face Hub repository id.
        verbose: whether to print verbose output. Defaults to `False`.
        disable_cuda_device_placement: whether to disable CUDA device placement. Defaults to `True`.
        normalize_embeddings: whether to normalize the embeddings. Defaults to `False`.
        extra_kwargs: additional dictionary of keyword arguments that will be passed to the
            `Llama` class of `llama_cpp` library. Defaults to `{}`.
        _model: the `Llama` model instance. This attribute is meant to be used internally
            and should not be accessed directly. It will be set in the `load` method.

    References:
        - [Offline inference embeddings](https://llama-cpp-python.readthedocs.io/en/stable/#embeddings)

    Examples:
        Generating sentence embeddings:

        ```python
        from distilabel.embeddings import LlamaCppEmbeddings

        embeddings = LlamaCppEmbeddings(model="/path/to/model.gguf")

        ## Hugging Face Hub

        ## embeddings = LlamaCppEmbeddings(hub_repository_id="second-state/All-MiniLM-L6-v2-Embedding-GGUF", model="all-MiniLM-L6-v2-Q2_K.gguf")

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        # [
        #   [-0.05447685346007347, -0.01623094454407692, ...],
        #   [4.4889533455716446e-05, 0.044016145169734955, ...],
        # ]
        ```
    """

    model: RuntimeParameter[str] = Field(
        default=None,
        description="Contains the path to the GGUF quantized model, compatible with the installed version of the `llama.cpp` Python bindings.",
    )
    hub_repository_id: RuntimeParameter[Union[None, str]] = Field(
        default=None,
        description="The Hugging Face Hub repository id.",
    )
    disable_cuda_device_placement: RuntimeParameter[bool] = Field(
        default=True,
        description="Whether to disable CUDA device placement.",
    )
    verbose: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to print verbose output from llama.cpp library.",
    )
    extra_kwargs: RuntimeParameter[Dict[str, Any]] = Field(
        default={},
        description="Additional dictionary of keyword arguments that will be passed to the `Llama` class of `llama_cpp` library.",
    )
    normalize_embeddings: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to normalize the embeddings.",
    )
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
            try:
                from huggingface_hub.utils import validate_repo_id

                validate_repo_id(self.hub_repository_id)
            except ImportError as ie:
                raise ImportError(
                    "Llama.from_pretrained requires the huggingface-hub package. "
                    "You can install it with `pip install huggingface-hub`."
                ) from ie
            try:
                self._logger.info(
                    f"Attempting to load model from Hugging Face Hub: {self.hub_repository_id}"
                )
                self._model = _LlamaCpp.from_pretrained(
                    repo_id=self.hub_repository_id,
                    filename=self.model,
                    verbose=self.verbose,
                    embedding=True,
                    kwargs=self.extra_kwargs,
                )
                self._logger.info("Model loaded successfully from Hugging Face Hub")
            except Exception as e:
                self._logger.error(
                    f"Failed to load model from Hugging Face Hub: {str(e)}"
                )
                raise
        else:
            try:
                self._logger.info(f"Attempting to load model from: {self.model_name}")
                self._model = _LlamaCpp(
                    model_path=self.model_name,
                    verbose=self.verbose,
                    embedding=True,
                    kwargs=self.extra_kwargs,
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
            embeds = self._model.embed(inputs, normalize=self.normalize_embeddings)
            return embeds
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            raise
