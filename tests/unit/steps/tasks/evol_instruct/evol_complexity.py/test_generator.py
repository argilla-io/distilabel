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

from distilabel.models.llms.base import LLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.evol_instruct.evol_complexity.generator import (
    EvolComplexityGenerator,
)
from distilabel.steps.tasks.evol_instruct.evol_complexity.utils import (
    GENERATION_MUTATION_TEMPLATES,
)


class TestEvolComplexityGenerator:
    def test_mutation_templates(self, dummy_llm: LLM) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        task = EvolComplexityGenerator(
            name="task", llm=dummy_llm, num_instructions=2, pipeline=pipeline
        )
        assert task.name == "task"
        assert task.llm is dummy_llm
        assert task.num_instructions == 2
        assert task.mutation_templates == GENERATION_MUTATION_TEMPLATES
        assert "BREADTH" not in task.mutation_templates
