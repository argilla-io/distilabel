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

import pytest

from distilabel.llms import SGLang


@pytest.fixture
def sglang_instance():
    return SGLang(model="test-model")


def test_sglang_init():
    """Test the initialization of SGLang class."""
    llm = SGLang(model="test-model")
    assert llm.model == "test-model"
    assert llm.dtype == "auto"
    assert llm.quantization is None


def test_sglang_load():
    """Test the load method of SGLang class."""
    llm = SGLang(model="meta-llama/Llama-2-7b-chat-hf")
    llm.load()
    assert llm._model is not None
    assert llm._tokenizer is not None


# Add more tests as needed for other methods and edge cases
