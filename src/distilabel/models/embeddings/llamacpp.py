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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import Field, PrivateAttr

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.embeddings.base import Embeddings
from distilabel.models.mixins.cuda_device_placement import CudaDevicePlacementMixin

if TYPE_CHECKING:
    from llama_cpp import Llama


class LlamaCppEmbeddings(Embeddings, CudaDevicePlacementMixin):
    """`LlamaCpp` library implementation for embedding generation.

    Attributes:
        model_name: contains the name of the GGUF quantized model, compatible with the
            installed version of the `llama.cpp` Python bindings.
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
        from distilabel.models.embeddings import LlamaCppEmbeddings

        # You can follow along this example downloading the following model running the following
        # command in the terminal, that will download the model to the `Downloads` folder:
        # curl -L -o ~/Downloads/all-MiniLM-L6-v2-Q2_K.gguf https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q2_K.gguf

        model_path = "Downloads/"
        model = "all-MiniLM-L6-v2-Q2_K.gguf"
        embeddings = LlamaCppEmbeddings(
            model=model,
            model_path=str(Path.home() / model_path),
        )

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        print(results)
        embeddings.unload()
        ```

        Generate sentence embeddings using a HuggingFace Hub model:

        ```python
        from distilabel.models.embeddings import LlamaCppEmbeddings
        # You need to set environment variable to download private model to the local machine

        repo_id = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"
        model = "all-MiniLM-L6-v2-Q2_K.gguf"
        embeddings = LlamaCppEmbeddings(model=model,repo_id=repo_id)

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        print(results)
        embeddings.unload()
        # [
        #   [-0.05447685346007347, -0.01623094454407692, ...],
        #   [4.4889533455716446e-05, 0.044016145169734955, ...],
        # ]
        ```

        Generate sentence embeddings with cpu:

        ```python
        from pathlib import Path
        from distilabel.models.embeddings import LlamaCppEmbeddings

        # You can follow along this example downloading the following model running the following
        # command in the terminal, that will download the model to the `Downloads` folder:
        # curl -L -o ~/Downloads/all-MiniLM-L6-v2-Q2_K.gguf https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q2_K.gguf

        model_path = "Downloads/"
        model = "all-MiniLM-L6-v2-Q2_K.gguf"
        embeddings = LlamaCppEmbeddings(
            model=model,
            model_path=str(Path.home() / model_path),
            n_gpu_layers=0,
            disable_cuda_device_placement=True,
        )

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        print(results)
        embeddings.unload()
        # [
        #   [-0.05447685346007347, -0.01623094454407692, ...],
        #   [4.4889533455716446e-05, 0.044016145169734955, ...],
        # ]
        ```


    """

    model: str = Field(
        description="The name of the model to use for embeddings.",
    )

    model_path: RuntimeParameter[str] = Field(
        default=None,
        description="The path to the GGUF quantized model, compatible with the installed version of the `llama.cpp` Python bindings.",
    )

    repo_id: RuntimeParameter[str] = Field(
        default=None, description="The Hugging Face Hub repository id.", exclude=True
    )

    n_gpu_layers: RuntimeParameter[int] = Field(
        default=-1,
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
        CudaDevicePlacementMixin.load(self)

        try:
            from llama_cpp import Llama
        except ImportError as ie:
            raise ImportError(
                "`llama-cpp-python` package is not installed. Please install it using"
                " `pip install 'distilabel[llama-cpp]'`."
            ) from ie

        if self.repo_id is not None:
            # use repo_id to download the model
            from huggingface_hub.utils import validate_repo_id

            validate_repo_id(self.repo_id)
            self._model = Llama.from_pretrained(
                repo_id=self.repo_id,
                filename=self.model,
                n_gpu_layers=self.n_gpu_layers,
                seed=self.seed,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                verbose=self.verbose,
                embedding=True,
                kwargs=self.extra_kwargs,
            )
        elif self.model_path is not None:
            self._model = Llama(
                model_path=str(Path(self.model_path) / self.model),
                n_gpu_layers=self.n_gpu_layers,
                seed=self.seed,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                verbose=self.verbose,
                embedding=True,
                kwargs=self.extra_kwargs,
            )
        else:
            raise ValueError("Either 'model_path' or 'repo_id' must be provided")

    def unload(self) -> None:
        """Unloads the `gguf` model."""
        CudaDevicePlacementMixin.unload(self)
        self._model.close()
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
        return self._model.embed(inputs, normalize=self.normalize_embeddings)
