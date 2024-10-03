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
import tempfile
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
        model_path: contains the path to the GGUF quantized model, compatible with the
            installed version of the `llama.cpp` Python bindings.
        repo_id: the Hugging Face Hub repository id.
        hf_token: Hugging Face token for accessing gated models.
        verbose: whether to print verbose output. Defaults to `False`.
        n_gpu_layers: number of layers to run on the GPU. Defaults to `-1` (use the GPU if available).
        disable_cuda_device_placement: whether to disable CUDA device placement. Defaults to `True`.
        normalize_embeddings: whether to normalize the embeddings. Defaults to `False`.
        seed: RNG seed, -1 for random
        n_ctx: Text context, 0 = from model
        n_batch: Prompt processing maximum batch size
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

        ## embeddings = LlamaCppEmbeddings(repo_id="second-state/All-MiniLM-L6-v2-Embedding-GGUF", model="all-MiniLM-L6-v2-Q2_K.gguf")

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        # [
        #   [-0.05447685346007347, -0.01623094454407692, ...],
        #   [4.4889533455716446e-05, 0.044016145169734955, ...],
        # ]
        ```
    """

    model_path: str
    repo_id: RuntimeParameter[Union[None, str]] = Field(
        default=None,
        description="The Hugging Face Hub repository id.",
    )
    hf_token: RuntimeParameter[Union[None, str]] = Field(
        default=None,
        description="Hugging Face token for accessing gated models.",
    )
    n_gpu_layers: int = 0
    disable_cuda_device_placement: RuntimeParameter[bool] = Field(
        default=True,
        description="Whether to disable CUDA device placement.",
    )
    verbose: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to print verbose output from llama.cpp library.",
    )
    normalize_embeddings: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to normalize the embeddings.",
    )
    seed: int = 4294967295
    n_ctx: int = 512
    n_batch: int = 512
    extra_kwargs: RuntimeParameter[Dict[str, Any]] = Field(
        default={},
        description="Additional dictionary of keyword arguments that will be passed to the `Llama` class of `llama_cpp` library.",
    )
    _model: Union["_LlamaCpp", None] = PrivateAttr(None)

    def load(self) -> None:
        """
        Loads the `gguf` model using either the path or the Hugging Face Hub repository id.
        If using Hugging Face Hub, the model will be downloaded to a local directory
        specified by the DISTILABEL_MODEL_DIR environment variable or to a temporary directory.
        """
        super().load()

        CudaDevicePlacementMixin.load(self)

        try:
            from llama_cpp import Llama as _LlamaCpp
        except ImportError as ie:
            raise ImportError(
                "`llama-cpp-python` package is not installed. Please install it using"
                " `pip install llama-cpp-python`."
            ) from ie

        if self.repo_id is not None:
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import validate_repo_id
            except ImportError as ie:
                raise ImportError(
                    "Llama.from_pretrained requires the huggingface-hub package. "
                    "You can install it with `pip install huggingface-hub`."
                ) from ie

            validate_repo_id(self.repo_id)

            # Determine the download directory
            download_dir = os.environ.get("DISTILABEL_MODEL_DIR")
            if download_dir is None:
                download_dir = tempfile.gettempdir()

            self._logger.info(
                f"Attempting to download model from Hugging Face Hub: {self.repo_id}"
            )
            try:
                model_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.model_path,
                    token=self.hf_token,
                    local_dir=download_dir,
                )
                self._logger.info(f"Model downloaded successfully to: {model_path}")
            except Exception as e:
                self._logger.error(
                    f"Failed to download model from Hugging Face Hub: {str(e)}"
                )
                raise

            try:
                self._model = _LlamaCpp(
                    model_path=model_path,
                    n_gpu_layers=self.n_gpu_layers,
                    seed=self.seed,
                    n_ctx=self.n_ctx,
                    n_batch=self.n_batch,
                    verbose=self.verbose,
                    embedding=True,
                    **self.extra_kwargs,
                )
                self._logger.info("Model loaded successfully")
            except Exception as e:
                self._logger.error(f"Failed to load model: {str(e)}")
                raise
        else:
            try:
                self._logger.info(f"Attempting to load model from: {self.model_path}")
                self._model = _LlamaCpp(
                    model_path=self.model_path,
                    seed=self.seed,
                    n_gpu_layers=self.n_gpu_layers,
                    n_ctx=self.n_ctx,
                    n_batch=self.n_batch,
                    verbose=self.verbose,
                    embedding=True,
                    **self.extra_kwargs,
                )
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
        return self.model_path

    def encode(self, inputs: List[str]) -> List[List[Union[int, float]]]:
        """Generates embeddings for the provided inputs.

        Args:
            inputs: a list of texts for which an embedding has to be generated.

        Returns:
            The generated embeddings.
        """
        return self._model.embed(inputs, normalize=self.normalize_embeddings)
