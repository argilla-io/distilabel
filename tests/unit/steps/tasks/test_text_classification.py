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

import json
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pytest
from pydantic import ValidationError

from distilabel.steps.tasks.text_classification import TextClassification
from tests.unit.conftest import DummyAsyncLLM

if TYPE_CHECKING:
    from distilabel.models.llms.typing import GenerateOutput
    from distilabel.steps.tasks.typing import FormattedInput


class TextClassificationLLM(DummyAsyncLLM):
    is_multilabel: bool = False

    async def agenerate(  # type: ignore
        self, input: "FormattedInput", num_generations: int = 1
    ) -> "GenerateOutput":
        if self.is_multilabel:
            labels = ["label_0", "label_1", "label_2"]
        else:
            labels = "label"
        return {
            "generations": [
                json.dumps({"labels": labels}) for _ in range(num_generations)
            ],
            "statistics": {
                "input_tokens": [12] * num_generations,
                "output_tokens": [12] * num_generations,
            },
        }


class TestTextClassification:
    @pytest.mark.parametrize(
        "is_multilabel, context, examples, available_labels, default_label, query_title",
        [
            (False, "context", None, None, "Unclassified", "User Query"),
            (False, "", ["example"], ["label1", "label2"], "default", "User Query"),
            (
                False,
                "",
                ["example"],
                {"label1": "explanation 1", "label2": "explanation 2"},
                "default",
                "User Query",
            ),
            (
                True,
                "",
                ["example", "other example"],
                None,
                "default",
                "User Query",
            ),
        ],
    )
    def test_format_input(
        self,
        is_multilabel: bool,
        context: str,
        examples: Optional[List[str]],
        available_labels: Optional[Union[List[str], Dict[str, str]]],
        default_label: Optional[Union[str, List[str]]],
        query_title: str,
    ) -> None:
        task = TextClassification(
            llm=DummyAsyncLLM(),
            is_multilabel=is_multilabel,
            context=context,
            examples=examples,
            available_labels=available_labels,
            default_label=default_label,
            query_title=query_title,
        )
        task.load()

        result = task.format_input({"text": "SAMPLE_TEXT"})
        content = result[1]["content"]

        assert f'respond with "{default_label}"' in content
        assert "## User Query\n```\nSAMPLE_TEXT\n```" in content
        assert f'respond with "{default_label}"' in content
        if not is_multilabel:
            assert "Provide the label that best describes the text." in content
            assert '```\n{\n    "labels": "label"\n}\n```' in content
        else:
            assert (
                "Provide a list with the label or labels that best describe the text. Do not include any label that do not apply."
                in content
            )
            assert '```\n{\n    "labels": [' in content
        if available_labels:
            if isinstance(available_labels, list):
                assert 'Use the available labels to classify the user query:\navailable_labels = [\n    "label1",\n    "label2"\n]'
            if isinstance(available_labels, dict):
                assert 'Use the available labels to classify the user query:\navailable_labels = [\n    "label1",  # explanation 1\n    "label2",  # explanation 2\n]'

        if examples:
            assert (
                "## Examples\nHere are some examples to help you understand the task:\n- example\n"
                in content
            )
        else:
            assert "## Examples" not in content
        assert (
            f"Please classify the {query_title.lower()} by assigning the most appropriate labels."
            in content
        )
        assert f"## {query_title}" in content

    @pytest.mark.parametrize(
        "is_multilabel, expected",
        [
            (False, json.dumps({"labels": "label"})),
            (
                True,
                [
                    json.dumps({"labels": ["label_0"]}),
                    json.dumps({"labels": ["label_0", "label_1"]}),
                    json.dumps({"labels": ["label_0", "label_1", "label_2"]}),
                ],
            ),
        ],
    )
    def test_process(self, is_multilabel: bool, expected: str) -> None:
        task = TextClassification(
            llm=TextClassificationLLM(is_multilabel=is_multilabel),
            is_multilabel=is_multilabel,
            use_default_structured_output=True,
        )
        task.load()
        result = next(task.process([{"text": "SAMPLE_TEXT"}]))
        assert result[0]["text"] == "SAMPLE_TEXT"
        if is_multilabel:
            assert result[0]["labels"] in [
                json.loads(opt)["labels"] for opt in expected
            ]
            assert (
                result[0]["distilabel_metadata"]["raw_output_text_classification_0"]
                in expected
            )
        else:
            assert result[0]["labels"] == json.loads(expected)["labels"]
            assert (
                result[0]["distilabel_metadata"]["raw_output_text_classification_0"]
                == expected
            )

    def test_multilabel_error(self) -> None:
        with pytest.raises(
            ValidationError,
            match=r"Only one of \'is_multilabel\' for TextClassifiaction or \'n\' for TextClustering can be set at the same time.",
        ):
            TextClassification(llm=DummyAsyncLLM(), is_multilabel=True, n=2)
