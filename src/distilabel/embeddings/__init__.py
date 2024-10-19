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

from distilabel.embeddings.base import Embeddings
from distilabel.embeddings.llamacpp import LlamaCppEmbeddings
from distilabel.embeddings.sentence_transformers import SentenceTransformerEmbeddings
from distilabel.embeddings.vllm import vLLMEmbeddings

__all__ = [
    "Embeddings",
    "SentenceTransformerEmbeddings",
    "vLLMEmbeddings",
    "LlamaCppEmbeddings",
]
