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

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import Field, PrivateAttr

from distilabel.embeddings.base import Embeddings
from distilabel.llms.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.mixins.runtime_parameters import RuntimeParameter

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Embeddings, CudaDevicePlacementMixin):
    """`sentence-transformers` library implementation for embedding generation.

    Attributes:
        model: the model Hugging Face Hub repo id or a path to a directory containing the
            model weights and configuration files.
        device: the name of the device used to load the model e.g. "cuda", "mps", etc.
            Defaults to `None`.
        prompts: a dictionary containing prompts to be used with the model. Defaults to
            `None`.
        default_prompt_name: the default prompt (in `prompts`) that will be applied to the
            inputs. If not provided, then no prompt will be used. Defaults to `None`.
        trust_remote_code: whether to allow fetching and executing remote code fetched
            from the repository in the Hub. Defaults to `False`.
        revision: if `model` refers to a Hugging Face Hub repository, then the revision
            (e.g. a branch name or a commit id) to use. Defaults to `"main"`.
        token: the Hugging Face Hub token that will be used to authenticate to the Hugging
            Face Hub. If not provided, the `HF_TOKEN` environment or `huggingface_hub` package
            local configuration will be used. Defaults to `None`.
        truncate_dim: the dimension to truncate the sentence embeddings. Defaults to `None`.
        model_kwargs: extra kwargs that will be passed to the Hugging Face `transformers`
            model class. Defaults to `None`.
        tokenizer_kwargs: extra kwargs that will be passed to the Hugging Face `transformers`
            tokenizer class. Defaults to `None`.
        config_kwargs: extra kwargs that will be passed to the Hugging Face `transformers`
            configuration class. Defaults to `None`.
        precision: the dtype that will have the resulting embeddings. Defaults to `"float32"`.
        normalize_embeddings: whether to normalize the embeddings so they have a length
            of 1. Defaults to `None`.

    Examples:
        Generating sentence embeddings:

        ```python
        from distilabel.embeddings import SentenceTransformerEmbeddings

        embeddings = SentenceTransformerEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")

        embeddings.load()

        results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
        # [
        #   [-0.05447685346007347, -0.01623094454407692, ...],
        #   [4.4889533455716446e-05, 0.044016145169734955, ...],
        # ]
        ```
    """

    model: str
    device: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The device to be used to load the model. If `None`, then it"
        " will check if a GPU can be used.",
    )
    prompts: Optional[Dict[str, str]] = None
    default_prompt_name: Optional[str] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    token: Optional[str] = None
    truncate_dim: Optional[int] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    config_kwargs: Optional[Dict[str, Any]] = None
    precision: Optional[Literal["float32", "int8", "uint8", "binary", "ubinary"]] = (
        "float32"
    )
    normalize_embeddings: RuntimeParameter[bool] = Field(
        default=True,
        description="Whether to normalize the embeddings so the generated vectors"
        " have a length of 1 or not.",
    )

    _model: Union["SentenceTransformer", None] = PrivateAttr(None)

    def load(self) -> None:
        """Loads the Sentence Transformer model"""
        super().load()

        if self.device == "cuda":
            CudaDevicePlacementMixin.load(self)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "`sentence-transformers` package is not installed. Please install it using"
                " `pip install sentence-transformers`."
            ) from e

        self._model = SentenceTransformer(
            model_name_or_path=self.model,
            device=self.device,
            prompts=self.prompts,
            default_prompt_name=self.default_prompt_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            token=self.token,
            truncate_dim=self.truncate_dim,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
        )

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
        return self._model.encode(  # type: ignore
            sentences=inputs,
            batch_size=len(inputs),
            convert_to_numpy=True,
            precision=self.precision,  # type: ignore
            normalize_embeddings=self.normalize_embeddings,  # type: ignore
        ).tolist()  # type: ignore

    def unload(self) -> None:
        del self._model
        if self.device == "cuda":
            CudaDevicePlacementMixin.unload(self)
        super().unload()
