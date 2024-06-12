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

import logging
from typing import Any, Dict, List, Union

import pytest
from _pytest.logging import LogCaptureFixture
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.benchmarks.arena_hard import ArenaHard, ArenaHardResults

from tests.unit.steps.tasks.utils import DummyLLM


class TestArenaHard:
    def test_format_input(self) -> None:
        task = ArenaHard(
            name="arena_hard",
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        result = task.format_input(
            input={
                "instruction": "INSTRUCTION",
                "generations": ["GENERATION_A", "GENERATION_B"],
            }
        )

        assert result[-1] == {
            "role": "user",
            "content": "<|User Prompt|>\nINSTRUCTION\n\n<|The Start of Assistant A's Answer|>\nGENERATION_A\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\nGENERATION_B\n<|The End of Assistant B's Answer|>",
        }

    @pytest.mark.parametrize(
        "output, expected",
        [
            (
                "My own answer to the prompt would be:\nANSWER\nMy final veredict is: [[A>>B]]\n",
                {
                    "evaluation": "My own answer to the prompt would be:\nANSWER\nMy final veredict is: [[A>>B]]\n",
                    "score": "A>>B",
                },
            ),
            (
                "My own answer to the prompt would be:\nANSWER\nMy final veredict is: TIE\n",
                {
                    "evaluation": "My own answer to the prompt would be:\nANSWER\nMy final veredict is: TIE\n",
                    "score": None,
                },
            ),
            (
                None,
                {"evaluation": None, "score": None},
            ),
        ],
    )
    def test_format_output(
        self, output: Union[str, None], expected: Dict[str, Any]
    ) -> None:
        task = ArenaHard(
            name="arena_hard",
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert (
            task.format_output(
                output=output,
                input={
                    "instruction": "INSTRUCTION",
                    "generations": ["GENERATION_A", "GENERATION_B"],
                },
            )
            == expected
        )


class TestArenaHardResults:
    @pytest.mark.parametrize(
        "custom_model_column, inputs",
        [
            ("model_name", ["evaluation", "score", "model_name"]),
            (None, ["evaluation", "score"]),
        ],
    )
    def test_inputs(
        self, custom_model_column: Union[str, None], inputs: List[str]
    ) -> None:
        step = ArenaHardResults(
            name="arena_hard_results",
            custom_model_column=custom_model_column,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        assert step.inputs == inputs

    def test_process(self, caplog: LogCaptureFixture) -> None:
        step = ArenaHardResults(
            name="arena_hard_results",
            custom_model_column="model_names",
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        step.load()

        with caplog.at_level(logging.INFO):
            next(
                step.process(
                    [
                        {
                            "evaluation": "...",
                            "score": "A>>B",
                            "model_names": ["gpt-4-0314", "other-model"],
                        },
                        {
                            "evaluation": "...",
                            "score": "A=B",
                            "model_names": ["gpt-4-0314", "other-model"],
                        },
                        {
                            "evaluation": "...",
                            "score": "B>>A",
                            "model_names": ["gpt-4-0314", "other-model"],
                        },
                    ]
                )
            )
        assert (
            "Arena Hard ELO: other-model    1445.577347\ngpt-4-0314     1000.000000\ndtype: float64\n"
            in caplog.text
        )

    def test_process_errors(self) -> None:
        step = ArenaHardResults(
            name="arena_hard_results",
            custom_model_column="model_names",
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        step.load()

        with pytest.raises(
            ValueError,
            match="This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0.0",
        ):
            next(
                step.process(
                    [
                        {
                            "evaluation": "...",
                            "score": "A>>B",
                            "model_names": ["gpt-4-0314", "other-model"],
                        },
                        {
                            "evaluation": "...",
                            "score": "B>>A",
                            "model_names": ["gpt-4-0314", "other-model"],
                        },
                    ]
                )
            )
