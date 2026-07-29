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

from typing import Any, Dict, List

import pytest
from pydantic import PrivateAttr

from distilabel.errors import DistilabelNotImplementedError
from distilabel.models.llms.base import LLM
from tests.unit.conftest import DummyLLM


class _DummyLLMWithKwargs(LLM):
    _last_kwargs: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def load(self) -> None:
        super().load()

    @property
    def model_name(self) -> str:
        return "test-kwargs"

    def generate(  # type: ignore[override]
        self,
        inputs: List[Any],
        num_generations: int = 1,
        **kwargs: Any,
    ) -> List[Any]:
        self._last_kwargs = kwargs
        return [
            {
                "generations": ["output"] * num_generations,
                "statistics": {
                    "input_tokens": [0] * num_generations,
                    "output_tokens": [0] * num_generations,
                },
            }
        ] * len(inputs)


class TestLLM:
    def test_offline_batch_generate_raise_distilabel_not_implemented_error(
        self,
    ) -> None:
        llm = DummyLLM()

        with pytest.raises(DistilabelNotImplementedError):
            llm.offline_batch_generate()

    def test_generate_outputs_merges_generation_kwargs(self) -> None:
        llm = _DummyLLMWithKwargs(generation_kwargs={"max_new_tokens": 2048})
        llm.load()

        llm.generate_outputs(inputs=[[]])

        assert llm._last_kwargs == {"max_new_tokens": 2048}

    def test_generate_outputs_call_kwargs_override_generation_kwargs(self) -> None:
        llm = _DummyLLMWithKwargs(
            generation_kwargs={"max_new_tokens": 2048, "temperature": 0.5}
        )
        llm.load()

        llm.generate_outputs(inputs=[[]], max_new_tokens=100)

        assert llm._last_kwargs == {"max_new_tokens": 100, "temperature": 0.5}
