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

from distilabel.llms.anthropic import AnthropicLLM
from distilabel.llms.anyscale import AnyscaleLLM
from distilabel.llms.azure import AzureOpenAILLM
from distilabel.llms.base import LLM, AsyncLLM
from distilabel.llms.cohere import CohereLLM
from distilabel.llms.groq import GroqLLM
from distilabel.llms.huggingface import InferenceEndpointsLLM, TransformersLLM
from distilabel.llms.litellm import LiteLLM
from distilabel.llms.llamacpp import LlamaCppLLM
from distilabel.llms.mistral import MistralLLM
from distilabel.llms.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.llms.moa import MixtureOfAgentsLLM
from distilabel.llms.ollama import OllamaLLM
from distilabel.llms.oneai import OneAI
from distilabel.llms.openai import OpenAILLM
from distilabel.llms.together import TogetherLLM
from distilabel.llms.typing import GenerateOutput, HiddenState
from distilabel.llms.vertexai import VertexAILLM
from distilabel.llms.vllm import ClientvLLM, vLLM

__all__ = [
    "AnthropicLLM",
    "AnyscaleLLM",
    "AzureOpenAILLM",
    "LLM",
    "AsyncLLM",
    "CohereLLM",
    "GroqLLM",
    "InferenceEndpointsLLM",
    "LiteLLM",
    "LlamaCppLLM",
    "MistralLLM",
    "CudaDevicePlacementMixin",
    "MixtureOfAgentsLLM",
    "OllamaLLM",
    "OneAI",
    "OpenAILLM",
    "TogetherLLM",
    "TransformersLLM",
    "GenerateOutput",
    "HiddenState",
    "VertexAILLM",
    "ClientvLLM",
    "vLLM",
]
