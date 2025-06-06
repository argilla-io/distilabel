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


from distilabel.models.embeddings.base import Embeddings
from distilabel.models.embeddings.llamacpp import LlamaCppEmbeddings
from distilabel.models.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddings,
)
from distilabel.models.embeddings.sglang import SGLangEmbeddings
from distilabel.models.embeddings.vllm import vLLMEmbeddings
from distilabel.models.image_generation.base import (
    AsyncImageGenerationModel,
    ImageGenerationModel,
)
from distilabel.models.image_generation.huggingface.inference_endpoints import (
    InferenceEndpointsImageGeneration,
)
from distilabel.models.image_generation.openai import OpenAIImageGeneration
from distilabel.models.llms.anthropic import AnthropicLLM
from distilabel.models.llms.anyscale import AnyscaleLLM
from distilabel.models.llms.azure import AzureOpenAILLM
from distilabel.models.llms.base import LLM, AsyncLLM
from distilabel.models.llms.cohere import CohereLLM
from distilabel.models.llms.groq import GroqLLM
from distilabel.models.llms.huggingface import InferenceEndpointsLLM, TransformersLLM
from distilabel.models.llms.litellm import LiteLLM
from distilabel.models.llms.llamacpp import LlamaCppLLM
from distilabel.models.llms.mistral import MistralLLM
from distilabel.models.llms.mlx import MlxLLM
from distilabel.models.llms.moa import MixtureOfAgentsLLM
from distilabel.models.llms.ollama import OllamaLLM
from distilabel.models.llms.openai import OpenAILLM
from distilabel.models.llms.sglang import ClientSGLang, SGLang
from distilabel.models.llms.together import TogetherLLM
from distilabel.models.llms.vertexai import VertexAILLM
from distilabel.models.llms.vllm import ClientvLLM, vLLM
from distilabel.models.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.typing import GenerateOutput, HiddenState

__all__ = [
    "LLM",
    "AnthropicLLM",
    "AnyscaleLLM",
    "AsyncImageGenerationModel",
    "AsyncLLM",
    "AzureOpenAILLM",
    "ClientSGLang",
    "ClientvLLM",
    "CohereLLM",
    "CudaDevicePlacementMixin",
    "Embeddings",
    "GenerateOutput",
    "GroqLLM",
    "HiddenState",
    "ImageGenerationModel",
    "InferenceEndpointsImageGeneration",
    "InferenceEndpointsLLM",
    "LiteLLM",
    "LlamaCppEmbeddings",
    "LlamaCppLLM",
    "MistralLLM",
    "MixtureOfAgentsLLM",
    "MlxLLM",
    "OllamaLLM",
    "OpenAIImageGeneration",
    "OpenAILLM",
    "SGLang",
    "SGLangEmbeddings",
    "SentenceTransformerEmbeddings",
    "TogetherLLM",
    "TransformersLLM",
    "VertexAILLM",
    "vLLM",
    "vLLMEmbeddings",
]
