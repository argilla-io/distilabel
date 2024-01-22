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

from unittest.mock import Mock

import pytest
from distilabel.llm import OllamaLLM
from distilabel.tasks.text_generation.base import TextGenerationTask


@pytest.fixture(scope="module")
def mock_ollama_llm():
    task = TextGenerationTask()
    with pytest.raises(ValueError):
        OllamaLLM(
            model="test_model",
            task=task,
        )
    OllamaLLM._api_available = Mock(return_value=None)
    with pytest.raises(ValueError):
        OllamaLLM(
            model="test_model",
            task=task,
        )
    OllamaLLM._api_model_available = Mock(return_value=None)
    OllamaLLM._text_generation_with_backoff = Mock(
        return_value={
            "model": "registry.ollama.ai/library/llama2:latest",
            "created_at": "2023-12-12T14:13:43.416799Z",
            "message": {"role": "assistant", "content": "Hello! How are you today?"},
            "done": True,
            "total_duration": 5191566416,
            "load_duration": 2154458,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 383809000,
            "eval_count": 298,
            "eval_duration": 4799921000,
        }
    )
    return OllamaLLM(
        model="test_model",
        task=task,
        max_new_tokens=10,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        num_threads=4,
        prompt_format="default",
        prompt_formatting_fn=None,
    )


def test_ollama_llm_init(mock_ollama_llm: OllamaLLM):
    assert mock_ollama_llm.model == "test_model"
    assert mock_ollama_llm.max_new_tokens == 10
    assert mock_ollama_llm.temperature == 0.8
    assert mock_ollama_llm.top_k == 5
    assert mock_ollama_llm.top_p == 0.9


def test_ollama_llm_inherits_from_task(mock_ollama_llm):
    assert isinstance(mock_ollama_llm.task, TextGenerationTask)


def test_ollama_llm_generate(mock_ollama_llm):
    llm_output = mock_ollama_llm._generate([{"input": "Hello!"}] * 10)
    assert isinstance(llm_output, list)
    assert len(llm_output) == 10
    assert isinstance(llm_output[0], list)
    assert len(llm_output[0]) == 1
    assert isinstance(llm_output[0][0], dict)
    assert llm_output[0][0].get("raw_output") == "Hello! How are you today?"
