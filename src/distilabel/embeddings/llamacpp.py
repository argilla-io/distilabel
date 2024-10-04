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

from distilabel.embeddings.base import Embeddings
from distilabel.llms.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.mixins.runtime_parameters import RuntimeParameter

if TYPE_CHECKING:
    from llama_cpp import Llama


class LlamaCppEmbeddings(Embeddings, CudaDevicePlacementMixin):
    """`LlamaCpp` library implementation for embedding generation.

    Attributes:
        model_path: contains the path to the GGUF quantized model, compatible with the
            installed version of the `llama.cpp` Python bindings.
        repo_id: the Hugging Face Hub repository id.
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

    Runtime parameters:
        - `n_gpu_layers`: the number of layers to use for the GPU. Defaults to `-1`.
        - `verbose`: whether to print verbose output. Defaults to `False`.
        - `normalize_embeddings`: whether to normalize the embeddings. Defaults to `False`.
        - `extra_kwargs`: additional dictionary of keyword arguments that will be passed to the
            `Llama` class of `llama_cpp` library. Defaults to `{}`.
    References:
        - [Offline inference embeddings](https://llama-cpp-python.readthedocs.io/en/stable/#embeddings)

    Examples:
        Generate sentence embeddings using a local model:

        ```python
        from pathlib import Path
        from distilabel.embeddings import LlamaCppEmbeddings

        # You can follow along this example downloading the following model running the following
        # command in the terminal, that will download the model to the `Downloads` folder:
        # curl -L -o ~/Downloads/All-MiniLM-L6-v2-Embedding-GGUF https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/blob/main/all-MiniLM-L6-v2-Q2_K.gguf

        model_path = "Downloads/all-MiniLM-L6-v2-Q2_K.gguf"
        embeddings = LlamaCppEmbeddings(model_path=str(Path.home() / model_path))

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        # [
        #   [-0.05447685346007347, -0.01623094454407692, ...],
        #   [4.4889533455716446e-05, 0.044016145169734955, ...],
        # ]
        ```

        Generate sentence embeddings using a HuggingFace Hub public model:

        ```python
        from pathlib import Path
        from distilabel.embeddings import LlamaCppEmbeddings

        repo_id = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"
        model_path = "all-MiniLM-L6-v2-Q5_K_M.gguf"
        embeddings = LlamaCppEmbeddings(repo_id=repo_id,model_path=model_path)

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        # [
        #   [-0.05447685346007347, -0.01623094454407692, ...],
        #   [4.4889533455716446e-05, 0.044016145169734955, ...],
        # ]
        ```

        Generate sentence embeddings using a HuggingFace Hub private model:

        ```python
        from pathlib import Path
        from distilabel.embeddings import LlamaCppEmbeddings

        # You need to set environment variable to download private model to the local machine
        os.environ["HF_TOKEN"] = "hf_..."

        repo_id = "private_repo_id"
        model_path = "model"
        embeddings = LlamaCppEmbeddings(repo_id=repo_id,model_path=model_path)

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        # [
        #   [-0.05447685346007347, -0.01623094454407692, ...],
        #   [4.4889533455716446e-05, 0.044016145169734955, ...],
        # ]
        ```



    """

    model_path: str

    repo_id: RuntimeParameter[str] = Field(
        default=None, description="The Hugging Face Hub repository id.", exclude=True
    )

    n_gpu_layers: RuntimeParameter[int] = Field(
        default=0,
        description="The number of layers that will be loaded in the GPU.",
    )

    n_ctx: int = 512
    n_batch: int = 512
    seed: int = 4294967295

    normalize_embeddings: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to normalize the embeddings.",
    )
    verbose: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to print verbose output from llama.cpp library.",
    )
    extra_kwargs: Optional[RuntimeParameter[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Additional dictionary of keyword arguments that will be passed to the"
        " `Llama` class of `llama_cpp` library. See all the supported arguments at: "
        "https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__",
    )
    _model: Optional["Llama"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the `gguf` model using either the path or the Hugging Face Hub repository id."""
        super().load()
        self.disable_cuda_device_placement = True
        CudaDevicePlacementMixin.load(self)

        try:
            from llama_cpp import Llama
        except ImportError as ie:
            raise ImportError(
                "`llama-cpp-python` package is not installed. Please install it using"
                " `pip install llama-cpp-python`."
            ) from ie

        if self.repo_id is not None:
            try:
                from huggingface_hub.utils import validate_repo_id

                validate_repo_id(self.repo_id)
            except ImportError as ie:
                raise ImportError(
                    "Llama.from_pretrained requires the huggingface-hub package. "
                    "You can install it with `pip install huggingface-hub`."
                ) from ie
            try:
                self._model = Llama.from_pretrained(
                    repo_id=self.repo_id,
                    filename=self.model_path,
                    n_gpu_layers=self.n_gpu_layers,
                    seed=self.seed,
                    n_ctx=self.n_ctx,
                    n_batch=self.n_batch,
                    verbose=self.verbose,
                    embedding=True,
                    kwargs=self.extra_kwargs,
                )
            except Exception:
                raise
        else:
            try:
                self._model = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=self.n_gpu_layers,
                    seed=self.seed,
                    n_ctx=self.n_ctx,
                    n_batch=self.n_batch,
                    verbose=self.verbose,
                    embedding=True,
                    kwargs=self.extra_kwargs,
                )
            except Exception:
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
