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

from distilabel.steps.tasks.apigen.semantic_checker import APIGenSemanticChecker
from tests.unit.conftest import DummySyncLLM

SAMPLE_DATA = [
    # The info can for the function description can be obtained from the tool itself
    {
        "func_desc": "Fetch information about a specific cat breed from the Cat Breeds API.",
        "query": "What information can be obtained about the Maine Coon cat breed?",
        "answers": '[{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}]',
        "execution_result": "Hopefully some info about the Maine Coon",
    },
    {
        "func_desc": "Checks if an email domain is valid or a disposable/temporary address.",
        "query": "Check if the email domains 'protonmail.com' and 'mail.com' are valid and not temporary. Get the products from category 'furniture' in my store, skipping the first 20 items and limiting to 25 items.",
        "answers": '[{"name": "mailcheck", "arguments": {"domain": "protonmail.com"}}, {"name": "mailcheck", "arguments": {"domain": "mail.com"}}, {"name": "get_products_in_category", "arguments": {"skip": 20, "limit": 25, "category": "furniture"}}]',
        "execution_result": "Response for the emails",
    },
    {
        "func_desc": "Fetches the content of a node in a navigation hierarchy.",
        "query": "What are the node contents for category IDs 8899 and 7766 in English and for category IDs 5544 and 3322 in French?",
        "answers": '[{"name": "navigations_get_node_content", "arguments": {"is_id": 8899, "cat_id": 8899, "language": "en"}}, {"name": "navigations_get_node_content", "arguments": {"is_id": 7766, "cat_id": 7766, "language": "en"}}, {"name": "navigations_get_node_content", "arguments": {"is_id": 5544, "cat_id": 5544, "language": "fr"}}, {"name": "navigations_get_node_content", "arguments": {"is_id": 3322, "cat_id": 3322, "language": "fr"}}]',
        "execution_result": "Response for the node contents",
    },
]


class TestAPIGenSemanticChecker:
    @pytest.mark.parametrize("use_default_structured_output", [True, False])
    def test_format_input(self, use_default_structured_output: bool) -> None:
        task = APIGenSemanticChecker(
            llm=DummySyncLLM(),
            use_default_structured_output=use_default_structured_output,
        )
        task.load()
        result = task.format_input(SAMPLE_DATA[0])
        assert isinstance(result, list)
        formatted_prompt = result[1]["content"]

        default_structured_output_check = "Your response MUST strictly adhere to the following JSON format, and NO other text MUST be included"
        if use_default_structured_output:
            assert default_structured_output_check not in formatted_prompt
        else:
            assert default_structured_output_check in formatted_prompt
        assert (
            '- Generated Function Calls: [{"name": "get_breed_information", "arguments": {"breed": "Maine Coon"}}]'
            in formatted_prompt
        )
        assert (
            "- All Available Functions:\nFetch information about a specific cat breed from the Cat Breeds API."
            in formatted_prompt
        )
        assert (
            "- Execution Results: Hopefully some info about the Maine Coon"
            in formatted_prompt
        )

    @pytest.mark.parametrize(
        "result, expected",
        [
            (
                '{"thought": "thought", "keep_row_after_semantic_check": "no", "passes": "no"}',
                {"thought": "thought", "keep_row_after_semantic_check": False},
            ),
            (None, {"thought": None, "keep_row_after_semantic_check": None}),
            ("wrong", {"thought": None, "keep_row_after_semantic_check": None}),
        ],
    )
    def test_format_output(self, result: str, expected: Dict[str, Any]) -> None:
        task = APIGenSemanticChecker(llm=DummySyncLLM())
        task.load()
        assert task.format_output(result, SAMPLE_DATA[0]) == expected
