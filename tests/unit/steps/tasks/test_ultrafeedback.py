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

from distilabel.llms.base import LLM
from distilabel.llms.typing import GenerateOutput
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.tasks.ultrafeedback import UltraFeedback


class UltraFeedbackLLM(LLM):
    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "ultrafeedback-model"

    def generate(
        self, inputs: List[ChatType], num_generations: int = 1, **kwargs: Any
    ) -> List[GenerateOutput]:
        return [
            [
                "Type: 1\nRationale: text\nRating: 1\nRationale: text\n\nType: 2\nRationale: text\nRating: 2\nRationale: text"
                for _ in range(num_generations)
            ]
            for _ in inputs
        ]


class TestUltraFeedback:
    def test_process_with_simple_aspect(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = UltraFeedbackLLM()

        task = UltraFeedback(
            name="ultrafeedback",
            aspect="instruction-following",
            llm=llm,
            pipeline=pipeline,
        )
        task.load()

        assert next(
            task.process([{"instruction": "test", "generations": ["A", "B"]}])
        ) == [
            {
                "instruction": "test",
                "generations": ["A", "B"],
                "ratings": [1, 2],
                "rationales": ["text", "text"],
                "model_name": "ultrafeedback-model",
                "distilabel_meta": {
                    "raw_output_ultrafeedback": "Type: 1\n"
                    "Rationale: text\n"
                    "Rating: 1\n"
                    "Rationale: text\n"
                    "\n"
                    "Type: 2\n"
                    "Rationale: text\n"
                    "Rating: 2\n"
                    "Rationale: text"
                },
            }
        ]

    def test_process_with_complex_aspect(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = UltraFeedbackLLM()

        task = UltraFeedback(
            name="ultrafeedback",
            aspect="truthfulness",
            llm=llm,
            pipeline=pipeline,
        )
        task.load()

        assert next(
            task.process([{"instruction": "test", "generations": ["A", "B"]}])
        ) == [
            {
                "instruction": "test",
                "generations": ["A", "B"],
                "types": [1, 2],
                "rationales": ["text", "text"],
                "ratings": [1, 2],
                "rationales-for-ratings": ["text", "text"],
                "model_name": "ultrafeedback-model",
                "distilabel_meta": {
                    "raw_output_ultrafeedback": "Type: 1\n"
                    "Rationale: text\n"
                    "Rating: 1\n"
                    "Rationale: text\n"
                    "\n"
                    "Type: 2\n"
                    "Rationale: text\n"
                    "Rating: 2\n"
                    "Rationale: text"
                },
            }
        ]
