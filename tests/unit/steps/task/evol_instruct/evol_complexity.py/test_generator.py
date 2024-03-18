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

from distilabel.llm.base import LLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.task.evol_instruct.evol_complexity.generator import (
    EvolComplexityGenerator,
)
from distilabel.steps.task.evol_instruct.evol_complexity.utils import (
    GenerationMutationTemplatesEvolComplexity,
)

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


class TestEvolComplexityGenerator:
    def test_mutation_templates(self):
        llm = DummyLLM()
        pipeline = Pipeline()
        task = EvolComplexityGenerator(
            name="task", llm=llm, num_instructions=2, pipeline=pipeline
        )
        assert task.name == "task"
        assert task.llm is llm
        assert task.num_instructions == 2
        assert task.mutation_templates == GenerationMutationTemplatesEvolComplexity
        assert task.generation_kwargs == {}
        assert "BREADTH" not in task.mutation_templates.__members__
