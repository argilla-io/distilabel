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

from typing import TYPE_CHECKING, Any, Dict, List, Union

import pytest
from distilabel.pipeline.local import Pipeline
from distilabel.steps.task.base import Task
from pydantic import ValidationError

from tests.unit.steps.task.utils import DummyLLM

if TYPE_CHECKING:
    from distilabel.steps.task.typing import ChatType


class DummyTask(Task):
    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": input["instruction"]},
        ]

    def format_output(self, output: Union[str, None], input: Dict[str, Any]) -> dict:
        return {"output": output}


class TestTask:
    def test_passing_pipeline(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = DummyTask(name="task", llm=llm, pipeline=pipeline)
        assert task.name == "task"
        assert task.llm is llm
        assert task.num_generations == 1
        assert task.group_generations is False
        assert task.generation_kwargs == {}
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self) -> None:
        with Pipeline() as pipeline:
            llm = DummyLLM()
            task = DummyTask(name="task", llm=llm, pipeline=pipeline)
            assert task.name == "task"
            assert task.llm is llm
            assert task.generation_kwargs == {}
        assert task.pipeline == pipeline

    def test_with_errors(self) -> None:
        with pytest.raises(ValueError, match="Step 'task' hasn't received a pipeline"):
            DummyTask(name="task", llm=DummyLLM())

        with pytest.raises(
            ValidationError, match="llm\n  Field required \\[type=missing"
        ):
            DummyTask(name="task", pipeline=Pipeline())  # type: ignore

        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class Task with abstract methods format_input, format_output",
        ):
            Task(name="task", llm=DummyLLM())  # type: ignore

    @pytest.mark.parametrize(
        "group_generations, expected",
        [
            (
                False,
                [
                    {"instruction": "test", "output": "output", "model_name": "test"},
                    {"instruction": "test", "output": "output", "model_name": "test"},
                    {"instruction": "test", "output": "output", "model_name": "test"},
                ],
            ),
            (
                True,
                [
                    {
                        "instruction": "test",
                        "output": ["output", "output", "output"],
                        "model_name": "test",
                    },
                ],
            ),
        ],
    )
    def test_process(
        self, group_generations: bool, expected: List[Dict[str, Any]]
    ) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = DummyTask(
            name="task",
            llm=llm,
            pipeline=pipeline,
            group_generations=group_generations,
            num_generations=3,
        )
        result = next(task.process([{"instruction": "test"}]))
        assert result == expected

    def test_serialization(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = DummyTask(name="task", llm=llm, pipeline=pipeline)
        assert task.dump() == {
            "name": "task",
            "input_mappings": {},
            "output_mappings": {},
            "input_batch_size": 50,
            "llm": {
                "type_info": {
                    "module": "tests.unit.steps.task.utils",
                    "name": "DummyLLM",
                }
            },
            "generation_kwargs": {},
            "group_generations": False,
            "num_generations": 1,
            "runtime_parameters_info": [
                {
                    "name": "num_generations",
                    "description": "The number of generations to be produced per input.",
                    "optional": True,
                },
                {
                    "name": "generation_kwargs",
                    "optional": True,
                    "description": "The kwargs to be propagated to either `generate` or `agenerate` methods within each `LLM`. Note that these kwargs will be specific to each LLM, and while some as `temperature` may be present on each `LLM`, some others may not, so read the `LLM.{generate,agenerate}` signatures in advance to see which kwargs are available.",
                },
            ],
            "type_info": {
                "module": "tests.unit.steps.task.test_base",
                "name": "DummyTask",
            },
        }

        with Pipeline() as pipeline:
            new_task = DummyTask.from_dict(task.dump())
            assert isinstance(new_task, DummyTask)
