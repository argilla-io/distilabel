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

import re
from unittest import mock

import pytest
from distilabel.llm.anyscale import AnyscaleLLM
from distilabel.tasks.text_generation.base import TextGenerationTask


@mock.patch("distilabel.llm.anyscale.OpenAI")
@mock.patch(
    "distilabel.llm.anyscale.OpenAILLM.available_models",
    ["HuggingFaceH4/zephyr-7b-beta"],
)
class TestAnyscaleLLM:
    def test_llm_anyscale(self, mock_anyscale):
        llm = AnyscaleLLM(
            model="HuggingFaceH4/zephyr-7b-beta",
            task=TextGenerationTask(),
            api_key="OPENAI_API_KEY",
        )
        assert isinstance(llm, AnyscaleLLM)
        assert llm.model_name == "HuggingFaceH4/zephyr-7b-beta"

    def test_unavailable_model(self, mock_anyscale):
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "Provided `model` is not available in your Anyscale account, available models are ['HuggingFaceH4/zephyr-7b-beta']"
            ),
        ):
            AnyscaleLLM(
                model="other_model",
                task=TextGenerationTask(),
                api_key="OPENAI_API_KEY",
            )
