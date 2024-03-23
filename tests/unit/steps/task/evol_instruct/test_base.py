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
from distilabel.steps.task.evol_instruct.base import (
    EvolInstruct,
)
from distilabel.steps.task.evol_instruct.utils import (
    MutationTemplates,
)
from pydantic import ValidationError


class TestEvolInstruct:
    def test_passing_pipeline(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline()
        task = EvolInstruct(
            name="task", llm=dummy_llm, num_evolutions=2, pipeline=pipeline
        )
        assert task.name == "task"
        assert task.llm is dummy_llm
        assert task.num_evolutions == 2
        assert task.mutation_templates == MutationTemplates
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self, dummy_llm: LLM) -> None:
        with Pipeline() as pipeline:
            task = EvolInstruct(
                name="task", llm=dummy_llm, num_evolutions=2, pipeline=pipeline
            )
            assert task.name == "task"
            assert task.llm is dummy_llm
        assert task.pipeline == pipeline

    def test_with_errors(self, dummy_llm: LLM) -> None:
        with pytest.raises(
            ValidationError, match="num_evolutions\n  Field required \\[type=missing"
        ):
            EvolInstruct(name="task", pipeline=Pipeline())  # type: ignore

        with pytest.raises(ValueError, match="Step 'task' hasn't received a pipeline"):
            EvolInstruct(name="task", llm=dummy_llm, num_evolutions=2)

    def test_process(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline()
        task = EvolInstruct(
            name="task", llm=dummy_llm, num_evolutions=2, pipeline=pipeline
        )
        assert list(task.process([{"instruction": "test"}])) == [
            [
                {
                    "instruction": "test",
                    "evolved_instruction": "output",
                    "model_name": "test",
                }
            ]
        ]

    def test_process_store_evolutions(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline()
        task = EvolInstruct(
            name="task",
            llm=dummy_llm,
            num_evolutions=2,
            store_evolutions=True,
            pipeline=pipeline,
        )
        assert list(task.process([{"instruction": "test"}])) == [
            [
                {
                    "instruction": "test",
                    "evolved_instructions": ["output", "output"],
                    "model_name": "test",
                }
            ]
        ]

    def test_process_generate_answers(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline()
        task = EvolInstruct(
            name="task",
            llm=dummy_llm,
            num_evolutions=2,
            generate_answers=True,
            pipeline=pipeline,
        )
        assert list(task.process([{"instruction": "test"}])) == [
            [
                {
                    "instruction": "test",
                    "evolved_instruction": "output",
                    "answer": "output",
                    "model_name": "test",
                }
            ]
        ]

    def test_serialization(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline()
        task = EvolInstruct(
            name="task", llm=dummy_llm, num_evolutions=2, pipeline=pipeline
        )
        assert task.dump() == {
            "name": "task",
            "input_mappings": task.input_mappings,
            "output_mappings": task.output_mappings,
            "input_batch_size": task.input_batch_size,
            "llm": {
                "generation_kwargs": {},
                "type_info": {
                    "module": "tests.unit.steps.task.conftest",
                    "name": "DummyLLM",
                },
            },
            "group_generations": task.group_generations,
            "num_generations": task.num_generations,
            "num_evolutions": task.num_evolutions,
            "store_evolutions": task.store_evolutions,
            "generate_answers": task.generate_answers,
            "mutation_templates": {
                "_type": "enum",
                "_enum_type": "str",
                "_name": task.mutation_templates.__name__,
                "_values": {
                    mutation.name: mutation.value  # type: ignore
                    for mutation in task.mutation_templates.__members__.values()  # type: ignore
                },
            },
            "seed": task.seed,
            "runtime_parameters_info": [
                {
                    "name": "llm",
                    "runtime_parameters_info": [
                        {
                            "name": "generation_kwargs",
                            "description": "The kwargs to be propagated to either `generate` or `agenerate` methods within each `LLM`.",
                            "keys": [],
                        }
                    ],
                },
                {
                    "name": "num_generations",
                    "optional": True,
                    "description": "The number of generations to be produced per input.",
                },
                {
                    "name": "seed",
                    "optional": True,
                    "description": "As `numpy` is being used in order to randomly pick a mutation method, then is nice to seed a random seed.",
                },
            ],
            "type_info": {
                "module": "distilabel.steps.task.evol_instruct.base",
                "name": "EvolInstruct",
            },
        }

        with Pipeline() as pipeline:
            new_task = EvolInstruct.from_dict(task.dump())
            assert isinstance(new_task, EvolInstruct)
