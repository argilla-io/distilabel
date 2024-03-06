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

from typing import TYPE_CHECKING, List

import pytest
from distilabel.llm.base import LLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.task.base import Task
from distilabel.steps.task.text_generation import TextGeneration
from pydantic import ValidationError

if TYPE_CHECKING:
    from distilabel.steps.task.typing import ChatType


class DummyLLM(LLM):
    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "test"

    def generate(self, inputs: List["ChatType"]) -> List[str]:
        return ["output" for _ in inputs]


class TestTextGeneration:
    def test_passing_pipeline(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = TextGeneration(name="task", llm=llm, pipeline=pipeline)
        assert task.name == "task"
        assert task.llm is llm
        assert task.generation_kwargs == {}
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self) -> None:
        with Pipeline() as pipeline:
            llm = DummyLLM()
            task = TextGeneration(name="task", llm=llm, pipeline=pipeline)
            assert task.name == "task"
            assert task.llm is llm
            assert task.generation_kwargs == {}
        assert task.pipeline == pipeline

    def test_with_errors(self) -> None:
        with pytest.raises(ValueError, match="Step 'task' hasn't received a pipeline"):
            TextGeneration(name="task", llm=DummyLLM())

        with pytest.raises(
            ValidationError, match="llm\n  Field required \\[type=missing"
        ):
            TextGeneration(name="task", pipeline=Pipeline())  # type: ignore

        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class Task with abstract methods format_input, format_output",
        ):
            Task(name="task", llm=DummyLLM())  # type: ignore

    def test_process(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = TextGeneration(name="task", llm=llm, pipeline=pipeline)
        assert list(task.process([{"instruction": "test"}])) == [
            [{"instruction": "test", "generation": "output", "model_name": "test"}]
        ]

    def test_serialization(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = TextGeneration(name="task", llm=llm, pipeline=pipeline)
        assert task.dump() == {
            "name": "task",
            "input_mappings": {},
            "output_mappings": {},
            "input_batch_size": 50,
            "llm": {
                "type_info": {
                    "module": "tests.unit.steps.task.test_text_generation",
                    "name": "DummyLLM",
                }
            },
            "generation_kwargs": {},
            "runtime_parameters_info": [
                {
                    "name": "generation_kwargs",
                    "optional": True,
                    "description": "The kwargs to be propagated to either `generate` or `agenerate` methods within each `LLM`. Note that these kwargs will be specific to each LLM, and while some as `temperature` may be present on each `LLM`, some others may not, so read the `LLM.{generate,agenerate}` signatures in advance to see which kwargs are available.",
                }
            ],
            "type_info": {
                "module": "distilabel.steps.task.text_generation",
                "name": "TextGeneration",
            },
        }

        with Pipeline() as pipeline:
            new_task = TextGeneration.from_dict(task.dump())
            assert isinstance(new_task, TextGeneration)
