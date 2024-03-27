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

from typing import Any, List

from distilabel.llm.base import LLM
from distilabel.llm.typing import GenerateOutput
from distilabel.pipeline.local import Pipeline
from distilabel.steps.task.instruction_backtranslation import InstructionBacktranslation
from distilabel.steps.task.typing import ChatType


class InstructionBacktranslationLLM(LLM):
    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "instruction-backtranslation-model"

    def generate(
        self, inputs: List[ChatType], num_generations: int = 1, **kwargs: Any
    ) -> List[GenerateOutput]:
        return [
            ["This is the reason. Score: 1" for _ in range(num_generations)]
            for _ in inputs
        ]


class TestInstructionBacktranslation:
    def test_process(self) -> None:
        task = InstructionBacktranslation(
            name="instruction-backtranslation",
            llm=InstructionBacktranslationLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert next(task.process([{"instruction": "test", "generation": "A"}])) == [
            {
                "instruction": "test",
                "generation": "A",
                "score": 1,
                "reason": "This is the reason.",
                "model_name": "instruction-backtranslation-model",
            }
        ]
