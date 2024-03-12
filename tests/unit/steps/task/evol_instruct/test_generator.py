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
from distilabel.steps.task.evol_instruct.generator import EvolInstructGenerator
from distilabel.steps.task.evol_instruct.utils import GenerationMutationTemplates
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
    def test_passing_pipeline(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = EvolInstructGenerator(
            name="task", llm=llm, num_instructions=2, pipeline=pipeline
        )
        assert task.name == "task"
        assert task.llm is llm
        assert task.num_instructions == 2
        assert task.mutation_templates == GenerationMutationTemplates
        assert task.generation_kwargs == {}
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self) -> None:
        with Pipeline() as pipeline:
            llm = DummyLLM()
            task = EvolInstructGenerator(
                name="task", llm=llm, num_instructions=2, pipeline=pipeline
            )
            assert task.name == "task"
            assert task.llm is llm
            assert task.generation_kwargs == {}
        assert task.pipeline == pipeline

    def test_with_errors(self) -> None:
        with pytest.raises(
            ValidationError, match="num_instructions\n  Field required \\[type=missing"
        ):
            EvolInstructGenerator(name="task", pipeline=Pipeline())  # type: ignore

        with pytest.raises(ValueError, match="Step 'task' hasn't received a pipeline"):
            EvolInstructGenerator(name="task", llm=DummyLLM(), num_instructions=2)

    def test_process(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = EvolInstructGenerator(
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

    def test_process_generate_answers(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = EvolInstructGenerator(
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

    def test_serialization(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = EvolInstructGenerator(
            name="task", llm=llm, num_instructions=2, pipeline=pipeline
        )
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
                "_name": "GenerationMutationTemplates",
                "_values": {
                    "FRESH_START": "Write one question or request containing one or more of the following words: <PROMPT>",
                    "COMPLICATE": "Rewrite #Given Prompt# to make it slightly more complicated, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n",
                    "ADD_CONSTRAINTS": "Add a few more constraints or requirements to #Given Prompt#, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n",
                    "DEEPEN": "Slightly increase the depth and breadth of #Given Prompt#, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n",
                    "CONCRETIZE": "Make #Given Prompt# slightly more concrete, and create #New Prompt#.\n#Given Prompt#:\n\n<PROMPT>\n",
                    "INCREASE_REASONING": "If #Given Prompt# can be solved with just a few simple thinking processes, rewrite it to explicitly request multi-step reasoning, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n",
                    "SWITCH_TOPIC": "Rewrite #Given Prompt# by switching the topic, keeping the domain and difficulty level similar, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n",
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
                "name": "EvolInstructGenerator",
            },
        }

        with Pipeline() as pipeline:
            new_task = EvolInstructGenerator.from_dict(task.dump())
            assert isinstance(new_task, EvolInstructGenerator)
