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
from distilabel.steps.tasks.base import Task
from pydantic import ValidationError

from tests.unit.steps.tasks.utils import DummyLLM

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


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
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyLLM()
        task = DummyTask(name="task", llm=llm, pipeline=pipeline)
        assert task.name == "task"
        assert task.llm is llm
        assert task.num_generations == 1
        assert task.group_generations is False
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self) -> None:
        with Pipeline(name="unit-test-pipeline") as pipeline:
            llm = DummyLLM()
            task = DummyTask(name="task", llm=llm, pipeline=pipeline)
            assert task.name == "task"
            assert task.llm is llm
        assert task.pipeline == pipeline

    def test_with_errors(self) -> None:
        with pytest.raises(ValueError, match="Step 'task' hasn't received a pipeline"):
            DummyTask(name="task", llm=DummyLLM())

        with pytest.raises(
            ValidationError, match="llm\n  Field required \\[type=missing"
        ):
            DummyTask(name="task", pipeline=Pipeline(name="unit-test-pipeline"))  # type: ignore

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
        pipeline = Pipeline(name="unit-test-pipeline")
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
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyLLM()
        task = DummyTask(name="task", llm=llm, pipeline=pipeline)
        assert task.dump() == {
            "name": "task",
            "input_mappings": {},
            "output_mappings": {},
            "input_batch_size": 50,
            "llm": {
                "generation_kwargs": {},
                "type_info": {
                    "module": "tests.unit.steps.tasks.utils",
                    "name": "DummyLLM",
                },
            },
            "group_generations": False,
            "num_generations": 1,
            "runtime_parameters_info": [
                {
                    "description": "The number of rows that will contain the batches processed by the step.",
                    "name": "input_batch_size",
                    "optional": True,
                },
                {
                    "name": "llm",
                    "runtime_parameters_info": [
                        {
                            "description": "The kwargs to be propagated to either `generate` or "
                            "`agenerate` methods within each `LLM`.",
                            "keys": [
                                {
                                    "name": "kwargs",
                                    "optional": False,
                                },
                            ],
                            "name": "generation_kwargs",
                        },
                    ],
                },
                {
                    "name": "num_generations",
                    "description": "The number of generations to be produced per input.",
                    "optional": True,
                },
            ],
            "type_info": {
                "module": "tests.unit.steps.tasks.test_base",
                "name": "DummyTask",
            },
        }

        with Pipeline(name="unit-test-pipeline") as pipeline:
            new_task = DummyTask.from_dict(task.dump())
            assert isinstance(new_task, DummyTask)
