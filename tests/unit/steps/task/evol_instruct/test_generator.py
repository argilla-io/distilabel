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


class TestEvolInstructGenerator:
    def test_passing_pipeline(self, task_params_generator) -> None:
        task_class, mutation_templates_class = task_params_generator
        pipeline = Pipeline()
        llm = DummyLLM()
        task = task_class(name="task", llm=llm, num_instructions=2, pipeline=pipeline)
        assert task.name == "task"
        assert task.llm is llm
        assert task.num_instructions == 2
        assert task.mutation_templates == mutation_templates_class
        assert task.generation_kwargs == {}
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self, task_class_generator) -> None:
        with Pipeline() as pipeline:
            llm = DummyLLM()
            task = task_class_generator(
                name="task", llm=llm, num_instructions=2, pipeline=pipeline
            )
            assert task.name == "task"
            assert task.llm is llm
            assert task.generation_kwargs == {}
        assert task.pipeline == pipeline

    def test_with_errors(self, task_class_generator) -> None:
        with pytest.raises(
            ValidationError, match="num_instructions\n  Field required \\[type=missing"
        ):
            task_class_generator(name="task", pipeline=Pipeline())  # type: ignore

        with pytest.raises(ValueError, match="Step 'task' hasn't received a pipeline"):
            task_class_generator(name="task", llm=DummyLLM(), num_instructions=2)

    def test_process(self, task_class_generator) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = task_class_generator(
            name="task",
            llm=llm,
            num_instructions=1,
            min_length=1,
            max_length=10,
            pipeline=pipeline,
        )
        assert list(task.process()) == [
            (
                [
                    {
                        "instruction": "output",
                        "model_name": "test",
                    }
                ],
                True,
            )
        ]

    def test_process_generate_answers(self, task_class_generator) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = task_class_generator(
            name="task",
            llm=llm,
            num_instructions=1,
            min_length=1,
            max_length=10,
            generate_answers=True,
            pipeline=pipeline,
        )
        assert list(task.process()) == [
            (
                [
                    {
                        "instruction": "output",
                        "answer": "output",
                        "model_name": "test",
                    }
                ],
                True,
            )
        ]

    def test_serialization(self, task_params_generator) -> None:
        task_class, mutation_templates_class = task_params_generator
        pipeline = Pipeline()
        llm = DummyLLM()
        task = task_class(name="task", llm=llm, num_instructions=2, pipeline=pipeline)
        assert task.dump() == {
            "name": "task",
            "input_mappings": {},
            "output_mappings": {},
            "batch_size": 50,
            "input_batch_size": 50,
            "llm": {
                "type_info": {
                    "module": "tests.unit.steps.task.evol_instruct.test_generator",
                    "name": "DummyLLM",
                }
            },
            "num_instructions": 2,
            "generate_answers": False,
            "mutation_templates": {
                "_type": "enum",
                "_enum_type": "str",
                "_name": mutation_templates_class.__name__,
                "_values": {
                    key: value.value
                    for key, value in mutation_templates_class.__members__.items()
                },
            },
            "min_length": 256,
            "max_length": 1024,
            "generation_kwargs": {},
            "runtime_parameters_info": [
                {
                    "name": "generation_kwargs",
                    "optional": True,
                    "description": "The kwargs to be propagated to either `generate` or `agenerate` methods within each `LLM`. Note that these kwargs will be specific to each LLM, and while some as `temperature` may be present on each `LLM`, some others may not, so read the `LLM.{generate,agenerate}` signatures in advance to see which kwargs are available.",
                },
                {
                    "name": "min_length",
                    "optional": True,
                },
                {
                    "name": "max_length",
                    "optional": True,
                },
            ],
            "type_info": {
                "module": "distilabel.steps.task.evol_instruct.generator",
                "name": task_class.__name__,
            },
        }

        with Pipeline() as pipeline:
            new_task = task_class.from_dict(task.dump())
            assert isinstance(new_task, task_class)
