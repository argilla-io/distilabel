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

from distilabel.llm.anthropic import AnthropicLLM
from distilabel.llm.anyscale import AnyscaleLLM
from distilabel.llm.base import LLM, AsyncLLM
from distilabel.llm.huggingface.inference_endpoints import InferenceEndpointsLLM
from distilabel.llm.huggingface.transformers import TransformersLLM
from distilabel.llm.litellm import LiteLLM
from distilabel.llm.llamacpp import LlamaCppLLM
from distilabel.llm.mistral import MistralLLM
from distilabel.llm.mixins import CudaDevicePlacementMixin
from distilabel.llm.openai import OpenAILLM
from distilabel.llm.together import TogetherLLM
from distilabel.llm.typing import GenerateOutput, HiddenState
from distilabel.llm.vertexai import VertexAILLM
from distilabel.llm.vllm import vLLM

__all__ = [
    "AnthropicLLM",
    "AnyscaleLLM",
    "AsyncLLM",
    "CudaDevicePlacementMixin",
    "GenerateOutput",
    "HiddenState",
    "InferenceEndpointsLLM",
    "LlamaCppLLM",
    "LLM",
    "LiteLLM",
    "MistralLLM",
    "OpenAILLM",
    "TogetherLLM",
    "TransformersLLM",
    "VertexAILLM",
    "vLLM",
]
