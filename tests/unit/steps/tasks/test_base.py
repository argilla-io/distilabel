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

import sys
from dataclasses import field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pytest
from distilabel.mixins.runtime_parameters import RuntimeParameter
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


class DummyRuntimeLLM(DummyLLM):
    runtime_parameter: RuntimeParameter[int]
    runtime_parameter_optional: Optional[RuntimeParameter[int]] = field(default=None)


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

    def test_with_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        DummyTask(name="task", llm=DummyLLM())
        assert "Step 'task' hasn't received a pipeline" in caplog.text

        with pytest.raises(
            ValidationError, match="llm\n  Field required \\[type=missing"
        ):
            DummyTask(name="task", pipeline=Pipeline(name="unit-test-pipeline"))  # type: ignore

        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class Task with abstract methods format_input, format_output"
            if sys.version_info < (3, 12)
            else "Can't instantiate abstract class Task without an implementation for abstract methods 'format_input', 'format_output'",
        ):
            Task(name="task", llm=DummyLLM())  # type: ignore

    @pytest.mark.parametrize(
        "group_generations, expected",
        [
            (
                False,
                [
                    {
                        "instruction": "test",
                        "output": "output",
                        "model_name": "test",
                        "distilabel_metadata": {"raw_output_task": "output"},
                    },
                    {
                        "instruction": "test",
                        "output": "output",
                        "model_name": "test",
                        "distilabel_metadata": {"raw_output_task": "output"},
                    },
                    {
                        "instruction": "test",
                        "output": "output",
                        "model_name": "test",
                        "distilabel_metadata": {"raw_output_task": "output"},
                    },
                ],
            ),
            (
                True,
                [
                    {
                        "instruction": "test",
                        "output": ["output", "output", "output"],
                        "model_name": "test",
                        "distilabel_metadata": [
                            {"raw_output_task": "output"},
                            {"raw_output_task": "output"},
                            {"raw_output_task": "output"},
                        ],
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

    def test_process_with_runtime_parameters(self) -> None:
        # 1. Runtime parameters provided
        llm = DummyRuntimeLLM()  # type: ignore
        task = DummyTask(
            name="task",
            llm=llm,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.set_runtime_parameters({"llm": {"runtime_parameter": 1}})
        assert task.load() is None
        assert task.llm.runtime_parameter == 1  # type: ignore
        assert task.llm.runtime_parameters_names == {
            "runtime_parameter": False,
            "runtime_parameter_optional": True,
            "generation_kwargs": {"kwargs": False},
        }

        # 2. Runtime parameters in init
        llm = DummyRuntimeLLM(runtime_parameter=1)
        task = DummyTask(
            name="task",
            llm=llm,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        assert task.load() is None
        assert task.llm.runtime_parameter == 1  # type: ignore
        assert task.llm.runtime_parameters_names == {
            "runtime_parameter": False,
            "runtime_parameter_optional": True,
            "generation_kwargs": {"kwargs": False},
        }

        # 3. Runtime parameters in init superseded by runtime parameters
        llm = DummyRuntimeLLM(runtime_parameter=1)
        task = DummyTask(
            name="task",
            llm=llm,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.set_runtime_parameters({"llm": {"runtime_parameter": 2}})
        assert task.load() is None
        assert task.llm.runtime_parameter == 2  # type: ignore
        assert task.llm.runtime_parameters_names == {
            "runtime_parameter": False,
            "runtime_parameter_optional": True,
            "generation_kwargs": {"kwargs": False},
        }

    def test_serialization(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyLLM()
        task = DummyTask(name="task", llm=llm, pipeline=pipeline)
        assert task.dump() == {
            "name": "task",
            "add_raw_output": True,
            "input_mappings": {},
            "output_mappings": {},
            "input_batch_size": 50,
            "llm": {
                "generation_kwargs": {},
                "structured_output": None,
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
                    "description": "Whether to include the raw output of the LLM in the output.",
                    "name": "add_raw_output",
                    "optional": True,
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
