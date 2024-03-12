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
from distilabel.steps.task.evol_instruct.base import EvolInstruct
from distilabel.steps.task.evol_instruct.utils import MutationTemplates
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


class TestEvolInstruct:
    def test_passing_pipeline(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = EvolInstruct(name="task", llm=llm, num_evolutions=2, pipeline=pipeline)
        assert task.name == "task"
        assert task.llm is llm
        assert task.num_evolutions == 2
        assert task.mutation_templates == MutationTemplates
        assert task.generation_kwargs == {}
        assert task.pipeline is pipeline

    def test_within_pipeline_context(self) -> None:
        with Pipeline() as pipeline:
            llm = DummyLLM()
            task = EvolInstruct(
                name="task", llm=llm, num_evolutions=2, pipeline=pipeline
            )
            assert task.name == "task"
            assert task.llm is llm
            assert task.generation_kwargs == {}
        assert task.pipeline == pipeline

    def test_with_errors(self) -> None:
        with pytest.raises(
            ValidationError, match="num_evolutions\n  Field required \\[type=missing"
        ):
            EvolInstruct(name="task", pipeline=Pipeline())  # type: ignore

        with pytest.raises(ValueError, match="Step 'task' hasn't received a pipeline"):
            EvolInstruct(name="task", llm=DummyLLM(), num_evolutions=2)

    def test_process(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = EvolInstruct(name="task", llm=llm, num_evolutions=2, pipeline=pipeline)
        assert list(task.process([{"instruction": "test"}])) == [
            [
                {
                    "instruction": "test",
                    "evolved_instruction": "output",
                    "model_name": "test",
                }
            ]
        ]

    def test_process_store_evolutions(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = EvolInstruct(
            name="task",
            llm=llm,
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

    def test_process_generate_answers(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = EvolInstruct(
            name="task",
            llm=llm,
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

    def test_serialization(self) -> None:
        pipeline = Pipeline()
        llm = DummyLLM()
        task = EvolInstruct(name="task", llm=llm, num_evolutions=2, pipeline=pipeline)
        assert task.dump() == {
            "name": "task",
            "input_mappings": {},
            "output_mappings": {},
            "input_batch_size": 50,
            "llm": {
                "type_info": {
                    "module": "tests.unit.steps.task.evol_instruct.test_base",
                    "name": "DummyLLM",
                }
            },
            "num_evolutions": 2,
            "store_evolutions": False,
            "generate_answers": False,
            "mutation_templates": {
                "_type": "enum",
                "_enum_type": "str",
                "_name": "MutationTemplates",
                "_values": {
                    "COMPLICATE": "Rewrite #Given Prompt# to make it slightly more complicated, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n",
                    "ADD_CONSTRAINTS": "Add a few more constraints or requirements to #Given Prompt#, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n",
                    "DEEPEN": "Slightly increase the depth and breadth of #Given Prompt#, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n",
                    "CONCRETIZE": "Make #Given Prompt# slightly more concrete, and create #New Prompt#.\n#Given Prompt#:\n\n<PROMPT>\n",
                    "INCREASE_REASONING": "If #Given Prompt# can be solved with just a few simple thinking processes, rewrite it to explicitly request multi-step reasoning, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n",
                    "SWITCH_TOPIC": "Rewrite #Given Prompt# by switching the topic, keeping the domain and difficulty level similar, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n",
                },
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
                "module": "distilabel.steps.task.evol_instruct.base",
                "name": "EvolInstruct",
            },
        }

        with Pipeline() as pipeline:
            new_task = EvolInstruct.from_dict(task.dump())
            assert isinstance(new_task, EvolInstruct)
