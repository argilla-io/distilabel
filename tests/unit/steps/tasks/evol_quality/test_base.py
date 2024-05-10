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
from distilabel.llms.base import LLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.evol_quality.base import (
    EvolQuality,
)
from distilabel.steps.tasks.evol_quality.utils import MUTATION_TEMPLATES
from pydantic import ValidationError


class TestEvolQuality:
    def test_with_errors(
        self, caplog: pytest.LogCaptureFixture, dummy_llm: LLM
    ) -> None:
        with pytest.raises(
            ValidationError, match="num_evolutions\n  Field required \\[type=missing"
        ):
            EvolQuality(name="task", pipeline=Pipeline(name="unit-test-pipeline"))  # type: ignore

        EvolQuality(name="task", llm=dummy_llm, num_evolutions=2)
        assert "Step 'task' hasn't received a pipeline" in caplog.text

    def test_process(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolQuality(
            name="task", llm=dummy_llm, num_evolutions=2, pipeline=pipeline
        )
        task.load()
        assert list(task.process([{"instruction": "test", "response": "mock"}])) == [
            [
                {
                    "instruction": "test",
                    "response": "mock",
                    "evolved_response": "output",
                    "model_name": "test",
                }
            ]
        ]

    def test_process_store_evolutions(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolQuality(
            name="task",
            llm=dummy_llm,
            num_evolutions=2,
            store_evolutions=True,
            pipeline=pipeline,
        )
        task.load()
        assert list(task.process([{"instruction": "test", "response": "mock"}])) == [
            [
                {
                    "instruction": "test",
                    "response": "mock",
                    "evolved_responses": ["output", "output"],
                    "model_name": "test",
                }
            ]
        ]

    def test_serialization(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolQuality(
            name="task", llm=dummy_llm, num_evolutions=2, pipeline=pipeline
        )
        task.load()
        assert task.dump() == {
            "name": "task",
            "add_raw_output": False,
            "input_mappings": task.input_mappings,
            "output_mappings": task.output_mappings,
            "input_batch_size": task.input_batch_size,
            "llm": {
                "generation_kwargs": {},
                "structured_output": None,
                "type_info": {
                    "module": task.llm.__module__,
                    "name": task.llm.__class__.__name__,
                },
            },
            "num_evolutions": task.num_evolutions,
            "store_evolutions": task.store_evolutions,
            "mutation_templates": MUTATION_TEMPLATES,
            "num_generations": task.num_generations,
            "group_generations": task.group_generations,
            "include_original_response": task.include_original_response,
            "seed": task.seed,
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
                    "description": "As `numpy` is being used in order to randomly pick a mutation method, then is nice to set a random seed.",
                },
            ],
            "type_info": {
                "module": task.__module__,
                "name": task.__class__.__name__,
            },
        }

        with Pipeline(name="unit-test-pipeline") as pipeline:
            new_task = EvolQuality.from_dict(task.dump())
            assert isinstance(new_task, EvolQuality)
