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

from typing import Any, Dict

import pytest
from distilabel.steps.tasks.sentence_transformers import (
    POSITIVE_NEGATIVE_SYSTEM_PROMPT,
    POSITIVE_SYSTEM_PROMPT,
    GenerateSentencePair,
    GenerationAction,
)

from tests.unit.steps.tasks.utils import DummyLLM


class TestGenerateSentencePair:
    @pytest.mark.parametrize(
        "action,triplet,system_prompt",
        [
            (
                "paraphrase",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(action_sentence="paraphrase"),
            ),
            (
                "paraphrase",
                False,
                POSITIVE_SYSTEM_PROMPT.format(action_sentence="paraphrase"),
            ),
            (
                "semantically-similar",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(
                    action_sentence="be semantically similar to"
                ),
            ),
            (
                "semantically-similar",
                False,
                POSITIVE_SYSTEM_PROMPT.format(
                    action_sentence="be semantically similar to"
                ),
            ),
            (
                "query",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(
                    action_sentence="be a query for"
                ),
            ),
            (
                "query",
                False,
                POSITIVE_SYSTEM_PROMPT.format(action_sentence="be a query for"),
            ),
        ],
    )
    def test_format_input(
        self, action: GenerationAction, triplet: bool, system_prompt: str
    ) -> None:
        task = GenerateSentencePair(llm=DummyLLM(), action=action, triplet=triplet)
        task.load()

        assert task.format_input({"anchor": "This is a unit test"}) == [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "## Anchor\n\nThis is a unit test\n"},
        ]

    @pytest.mark.parametrize(
        "output,triplet,expected",
        [
            (
                "## Positive\n\nThis is a paraphrase\n## Negative\n\nThis is not a paraphrase",
                True,
                {
                    "positive": "This is a paraphrase",
                    "negative": "This is not a paraphrase",
                },
            ),
            (
                "## Positive\n\nThis is a paraphrase",
                True,
                {"positive": "This is a paraphrase", "negative": None},
            ),
            (
                "## Positive\n\nThis is a paraphrase",
                False,
                {"positive": "This is a paraphrase"},
            ),
            (
                "random",
                False,
                {"positive": None},
            ),
        ],
    )
    def test_format_output(
        self, output: str, triplet: bool, expected: Dict[str, Any]
    ) -> None:
        task = GenerateSentencePair(
            llm=DummyLLM(), action="paraphrase", triplet=triplet
        )
        task.load()

        assert task.format_output(output) == expected
