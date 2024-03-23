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
from distilabel.llm.base import LLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.task.evol_instruct.generator import (
    EvolInstructGenerator,
)
from distilabel.steps.task.evol_instruct.utils import (
    GenerationMutationTemplates,
)
from pydantic import ValidationError


class TestEvolInstructGenerator:
    def test_passing_pipeline(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline()
        task = EvolInstructGenerator(
            name="task", llm=dummy_llm, num_instructions=2, pipeline=pipeline
        )
        assert task.name == "task"
        assert task.llm is dummy_llm
        assert task.num_instructions == 2
        assert task.mutation_templates == GenerationMutationTemplates
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self, dummy_llm: LLM) -> None:
        with Pipeline() as pipeline:
            task = EvolInstructGenerator(
                name="task", llm=dummy_llm, num_instructions=2, pipeline=pipeline
            )
            assert task.name == "task"
            assert task.llm is dummy_llm
        assert task.pipeline == pipeline

    def test_with_errors(self, dummy_llm: LLM) -> None:
        with pytest.raises(
            ValidationError, match="num_instructions\n  Field required \\[type=missing"
        ):
            EvolInstructGenerator(name="task", pipeline=Pipeline())  # type: ignore

        with pytest.raises(ValueError, match="Step 'task' hasn't received a pipeline"):
            EvolInstructGenerator(name="task", llm=dummy_llm, num_instructions=2)

    def test_process(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline()
        task = EvolInstructGenerator(
            name="task",
            llm=dummy_llm,
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

    def test_process_generate_answers(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline()
        task = EvolInstructGenerator(
            name="task",
            llm=dummy_llm,
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

    def test_serialization(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline()
        task = EvolInstructGenerator(
            name="task", llm=dummy_llm, num_instructions=2, pipeline=pipeline
        )
        assert task.dump() == {
            "name": "task",
            "llm": {
                "generation_kwargs": {},
                "type_info": {
                    "module": "tests.unit.steps.task.conftest",
                    "name": "DummyLLM",
                },
            },
            "input_mappings": task.input_mappings,
            "output_mappings": task.output_mappings,
            "batch_size": task.batch_size,
            "num_instructions": task.num_instructions,
            "generate_answers": task.generate_answers,
            "mutation_templates": {
                "_type": "enum",
                "_enum_type": "str",
                "_name": task.mutation_templates.__name__,
                "_values": {
                    mutation.name: mutation.value  # type: ignore
                    for mutation in task.mutation_templates
                },
            },
            "num_generations": task.num_generations,
            "group_generations": task.group_generations,
            "min_length": task.min_length,
            "max_length": task.max_length,
            "seed": task.seed,
            "runtime_parameters_info": [
                {
                    "name": "llm",
                    "runtime_parameters_info": [
                        {
                            "description": "The kwargs to be propagated to either `generate` or `agenerate` methods within each `LLM`.",
                            "keys": [],
                            "name": "generation_kwargs",
                        },
                    ],
                },
                {
                    "name": "num_generations",
                    "optional": True,
                    "description": "The number of generations to be produced per input.",
                },
                {
                    "name": "min_length",
                    "optional": True,
                    "description": "Defines the length (in bytes) that the generated instruction needs to be higher than, to be considered valid.",
                },
                {
                    "name": "max_length",
                    "optional": True,
                    "description": "Defines the length (in bytes) that the generated instruction needs to be lower than, to be considered valid.",
                },
                {
                    "name": "seed",
                    "optional": True,
                    "description": "As `numpy` is being used in order to randomly pick a mutation method, then is nice to seed a random seed.",
                },
            ],
            "type_info": {
                "module": EvolInstructGenerator.__module__,
                "name": EvolInstructGenerator.__name__,
            },
        }

        with Pipeline() as pipeline:
            new_task = EvolInstructGenerator.from_dict(task.dump())
            assert isinstance(new_task, EvolInstructGenerator)
