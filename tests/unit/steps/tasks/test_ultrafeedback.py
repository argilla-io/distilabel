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

from typing import Any, Dict, List, Union

import pytest

from distilabel.models.llms.base import LLM
from distilabel.steps.tasks.ultrafeedback import UltraFeedback
from distilabel.typing import ChatType, GenerateOutput


class UltraFeedbackLLM(LLM):
    structured_output: Any = None

    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "ultrafeedback-model"

    def generate(
        self, inputs: List[ChatType], num_generations: int = 1, **kwargs: Any
    ) -> List[GenerateOutput]:
        return [
            {
                "generations": [
                    "Type: 1\nRationale: text\nRating: 1\nRationale: text\n\nType: 2\nRationale: text\nRating: 2\nRationale: text"
                    for i in range(num_generations)
                ],
                "statistics": {
                    "input_tokens": [12] * num_generations,
                    "output_tokens": [12] * num_generations,
                },
            }
        ] * len(inputs)


class TestUltraFeedback:
    def test_process_with_simple_aspect(self) -> None:
        task = UltraFeedback(
            name="ultrafeedback",
            aspect="instruction-following",
            llm=UltraFeedbackLLM(),
            use_default_structured_output=False,
            add_raw_input=False,
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
                "distilabel_metadata": {
                    "raw_output_ultrafeedback": "Type: 1\nRationale: text\nRating: 1\nRationale: text\n\nType: 2\nRationale: text\nRating: 2\nRationale: text",
                    "statistics_ultrafeedback": {
                        "input_tokens": 12,
                        "output_tokens": 12,
                    },
                },
            }
        ]

    def test_process_with_complex_aspect(self) -> None:
        task = UltraFeedback(
            name="ultrafeedback",
            aspect="truthfulness",
            llm=UltraFeedbackLLM(),
            use_default_structured_output=False,
            add_raw_input=False,
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
                "distilabel_metadata": {
                    "raw_output_ultrafeedback": "Type: 1\nRationale: text\nRating: 1\nRationale: text\n\nType: 2\nRationale: text\nRating: 2\nRationale: text",
                    "statistics_ultrafeedback": {
                        "input_tokens": 12,
                        "output_tokens": 12,
                    },
                },
            }
        ]

    @pytest.mark.parametrize(
        "output, use_default_structured_output, aspect, expected",
        [
            (
                "{ \n   random\n}",
                True,
                "honesty",
                {"ratings": [None, None], "rationales": [None, None]},
            ),
            (
                '{ \n  "ratings": [\n    1,\n    5\n  ]\n ,\n  "rationales": [\n    "rationale1",\n    "rationale2"\n  ]}',
                True,
                "honesty",
                {"ratings": [1, 5], "rationales": ["rationale1", "rationale2"]},
            ),
            (
                "{ \n   random\n}",
                True,
                "helpfulness",
                {
                    "ratings": [None, None],
                    "rationales": [None, None],
                    "rationales-for-ratings": [None, None],
                    "types": [None, None],
                },
            ),
            (
                '{ \n  "ratings": [\n    1,\n    5\n  ]\n ,\n  "rationales": [\n    "rationale1",\n    "rationale2"\n  ], "rationales-for-ratings": [\n    "rationale1",\n    "rationale2"\n  ], "types": [\n    1,\n    2\n  ]}',
                True,
                "helpfulness",
                {
                    "ratings": [1, 5],
                    "rationales": ["rationale1", "rationale2"],
                    "rationales-for-ratings": ["rationale1", "rationale2"],
                    "types": [1, 2],
                },
            ),
        ],
    )
    def test_format_output(
        self,
        output: Union[str, None],
        use_default_structured_output: bool,
        aspect: str,
        expected: Dict[str, Any],
    ) -> None:
        task = UltraFeedback(
            llm=UltraFeedbackLLM(),
            aspect=aspect,
            use_default_structured_output=use_default_structured_output,
        )
        task.load()

        result = task.format_output(
            output=output,
            input={
                "instruction": "How much is 2+2?",
                "generations": ["4", "something weird"],
            },
        )

        assert result == expected

    def test_format_output_non_numeric_rating(self) -> None:
        """A non-numeric rating that isn't the `None`/`N/A` sentinel must degrade
        gracefully to `None` instead of raising `IndexError` (`re.findall` returns
        `[]`, which previously crashed the pipeline on a malformed response)."""
        honesty_task = UltraFeedback(
            llm=UltraFeedbackLLM(),
            aspect="honesty",
            use_default_structured_output=False,
        )
        honesty_task.load()
        assert honesty_task.format_output(
            output="Rating: high\nRationale: r1\n\nRating: low\nRationale: r2",
            input={"instruction": "x", "generations": ["a", "b"]},
        ) == {"ratings": [None, None], "rationales": ["r1", "r2"]}

        helpfulness_task = UltraFeedback(
            llm=UltraFeedbackLLM(),
            aspect="helpfulness",
            use_default_structured_output=False,
        )
        helpfulness_task.load()
        assert helpfulness_task.format_output(
            output="Type: good\nRationale: r1\nRating: high\nRationale: rr1",
            input={"instruction": "x", "generations": ["a"]},
        ) == {
            "types": [None],
            "rationales": ["r1"],
            "ratings": [None],
            "rationales-for-ratings": ["rr1"],
        }
