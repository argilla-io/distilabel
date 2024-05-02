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
import urllib.request
from typing import Generator

import pytest
from distilabel.llms.llamacpp import LlamaCppLLM


@pytest.fixture(scope="module")
def llm() -> Generator[LlamaCppLLM, None, None]:
    if not os.path.exists("tinyllama.gguf"):
        urllib.request.urlretrieve(
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
            "tinyllama.gguf",
        )

    llm = LlamaCppLLM(model_path="tinyllama.gguf", n_gpu_layers=0)  # type: ignore
    llm.load()

    yield llm


class TestLlamaCppLLM:
    def test_model_name(self, llm: LlamaCppLLM) -> None:
        assert llm.model_name == "tinyllama.gguf"

    def test_generate(self, llm: LlamaCppLLM) -> None:
        responses = llm.generate(
            inputs=[
                [{"role": "user", "content": "Hello, how are you?"}],
                [
                    {
                        "role": "user",
                        "content": "You're GPT2, you're old now but you still serves a purpose which is being used in unit tests.",
                    }
                ],
            ],
            num_generations=3,
        )

        assert len(responses) == 2
        assert len(responses[0]) == 3
