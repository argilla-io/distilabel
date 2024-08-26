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

from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.self_instruct import SelfInstruct
from tests.unit.conftest import DummyLLM


class TestSelfInstruct:
    def test_format_input(self) -> None:
        task = SelfInstruct(
            name="self_instruct",
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        input = task.format_input(input={"input": "test"})
        assert input == [
            {
                "role": "user",
                "content": '# Task Description\nDevelop 5 user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model\'s textual capabilities.\n\n# Criteria for Queries\nIncorporate a diverse range of verbs, avoiding repetition.\nEnsure queries are compatible with AI model\'s text generation functions and are limited to 1-2 sentences.\nDesign queries to be self-contained and standalone.\nBlend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.\nWrite each query on a separate line and avoid using numbered lists or bullet points.\n\n# AI Application\nAI assistant\n\n# Context\ntest\n\n# Output\n',
            }
        ]

    def test_format_output(self) -> None:
        task = SelfInstruct(
            name="self_instruct",
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        output = task.format_output(
            output="Instruction 1\n\nInstruction 2\n\nInstruction 3"
        )
        assert output == {
            "instructions": ["Instruction 1", "Instruction 2", "Instruction 3"]
        }
