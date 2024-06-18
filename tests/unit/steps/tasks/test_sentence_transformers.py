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
    CONTEXT_INTRO,
    POSITIVE_NEGATIVE_SYSTEM_PROMPT,
    POSITIVE_SYSTEM_PROMPT,
    GenerateSentencePair,
    GenerationAction,
)

from tests.unit.conftest import DummyLLM


class TestGenerateSentencePair:
    @pytest.mark.parametrize(
        "action,triplet,system_prompt",
        [
            (
                "paraphrase",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(
                    action_sentence="paraphrase", context=""
                ),
            ),
            (
                "paraphrase",
                False,
                POSITIVE_SYSTEM_PROMPT.format(action_sentence="paraphrase", context=""),
            ),
            (
                "semantically-similar",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(
                    action_sentence="be semantically similar to", context=""
                ),
            ),
            (
                "semantically-similar",
                False,
                POSITIVE_SYSTEM_PROMPT.format(
                    action_sentence="be semantically similar to", context=""
                ),
            ),
            (
                "query",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(
                    action_sentence="be a query for", context=""
                ),
            ),
            (
                "query",
                False,
                POSITIVE_SYSTEM_PROMPT.format(
                    action_sentence="be a query for", context=""
                ),
            ),
            (
                "answer",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(
                    action_sentence="be an answer for", context=""
                ),
            ),
            (
                "answer",
                False,
                POSITIVE_SYSTEM_PROMPT.format(
                    action_sentence="be an answer for", context=""
                ),
            ),
        ],
    )
    def test_format_input(
        self, action: GenerationAction, triplet: bool, system_prompt: str
    ) -> None:
        task = GenerateSentencePair(llm=DummyLLM(), action=action, triplet=triplet)
        task.load()
        content = "## Anchor\n\nThis is a unit test\n"
        assert task.format_input({"anchor": "This is a unit test"}) == [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    @pytest.mark.parametrize(
        "action,triplet,system_prompt",
        [
            (
                "paraphrase",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(
                    action_sentence="paraphrase", context=CONTEXT_INTRO
                ),
            ),
            (
                "paraphrase",
                False,
                POSITIVE_SYSTEM_PROMPT.format(
                    action_sentence="paraphrase", context=CONTEXT_INTRO
                ),
            ),
            (
                "semantically-similar",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(
                    action_sentence="be semantically similar to", context=CONTEXT_INTRO
                ),
            ),
            (
                "semantically-similar",
                False,
                POSITIVE_SYSTEM_PROMPT.format(
                    action_sentence="be semantically similar to", context=CONTEXT_INTRO
                ),
            ),
            (
                "query",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(
                    action_sentence="be a query for", context=CONTEXT_INTRO
                ),
            ),
            (
                "query",
                False,
                POSITIVE_SYSTEM_PROMPT.format(
                    action_sentence="be a query for", context=CONTEXT_INTRO
                ),
            ),
            (
                "answer",
                True,
                POSITIVE_NEGATIVE_SYSTEM_PROMPT.format(
                    action_sentence="be an answer for", context=CONTEXT_INTRO
                ),
            ),
            (
                "answer",
                False,
                POSITIVE_SYSTEM_PROMPT.format(
                    action_sentence="be an answer for", context=CONTEXT_INTRO
                ),
            ),
        ],
    )
    def test_format_input_with_context(
        self, action: GenerationAction, triplet: bool, system_prompt: str
    ) -> None:
        context = "This is your context."
        task = GenerateSentencePair(
            llm=DummyLLM(),
            action=action,
            triplet=triplet,
            context=context,
        )
        task.load()
        content = f"## Context\n\n{context}\n\n## Anchor\n\nThis is a unit test\n"
        # content = f"## Anchor\n\nThis is a unit test\n## Context\n\n{context}"
        assert task.format_input({"anchor": "This is a unit test"}) == [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
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
