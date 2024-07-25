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
    from sentence_transformers import SentenceTransformer


class SentenceTransformers(Embeddings, CudaDevicePlacementMixin):
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
    normalize_embeddings: RuntimeParameter[bool] = Field(
        default=True,
        description="Whether to normalize the embeddings so the generated vectors"
        " have a length of 1 or not.",
    )

    _model: Union["SentenceTransformer", None] = PrivateAttr(None)

    def load(self) -> None:
        """Loads the Sentence Transformer model"""
        super().load()

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
        return self._model.encode(  # type: ignore
            sentences=inputs,
            batch_size=len(inputs),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,  # type: ignore
        ).tolist()  # type: ignore
