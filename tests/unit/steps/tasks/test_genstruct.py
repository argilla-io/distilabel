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

from typing import Any, Dict, Optional, Union
from unittest.mock import MagicMock

import pytest
from jinja2 import Template

from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.genstruct import Genstruct
from tests.unit.conftest import DummyAsyncLLM


class TestGenstruct:
    def test_format_input(self) -> None:
        task = Genstruct(
            name="genstruct",
            llm=DummyAsyncLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        result = task.format_input(
            input={"title": "This is the title.\n", "content": "This is the content.\n"}
        )

        assert result == [
            {
                "role": "user",
                "content": "[[[Title]]] This is the title.\n[[[Content]]] This is the content.\n\nThe following is an interaction between a user and an AI assistant that is related to the above text.\n\n[[[User]]] ",
            }
        ]

    @pytest.mark.parametrize(
        "output, expected",
        [
            (
                "This is the instruction.\n[[[Assistant]]] This is the response.\n",
                {
                    "user": "This is the instruction.",
                    "assistant": "This is the response.",
                },
            ),
            (
                None,
                {"user": None, "assistant": None},
            ),
        ],
    )
    def test_format_output(
        self, output: Union[str, None], expected: Dict[str, Any]
    ) -> None:
        task = Genstruct(
            name="genstruct",
            llm=DummyAsyncLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert (
            task.format_output(
                output=output,
                input={
                    "title": "This is the title.\n",
                    "content": "This is the content.\n",
                },
            )
            == expected
        )

    def test_load_sets_default_chat_template(self) -> None:
        class MockLLM(DummyAsyncLLM):
            chat_template: Optional[str] = None

        mock_llm = MockLLM()
        task = Genstruct(
            name="genstruct",
            llm=mock_llm,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert (
            mock_llm.chat_template
            == "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )

    def test_load_preserves_custom_chat_template(self) -> None:
        class MockLLM(DummyAsyncLLM):
            chat_template: Optional[str] = "custom_template"

        mock_llm = MockLLM()
        task = Genstruct(
            name="genstruct",
            llm=mock_llm,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert mock_llm.chat_template == "custom_template"

    def test_load_overrides_tokenizer_chat_template_with_title(self) -> None:
        genstruct_tokenizer_chat_template = (
            "{% for message in messages %}{{ '[[[Title]]] ' + message['title'].strip() + '\n' }}"
            "{{ '[[[Content]]] ' + message['content'].strip() + '\n\n' + "
            "'The following is an interaction between a user and an AI assistant that is related to the above text.\n\n[[[User]]] ' }}{% endfor %}"
        )
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = genstruct_tokenizer_chat_template

        def mock_apply_chat_template(messages, **kwargs):
            tmpl = Template(mock_tokenizer.chat_template)
            return tmpl.render(messages=messages)

        mock_tokenizer.apply_chat_template = mock_apply_chat_template

        class MockLLM(DummyAsyncLLM):
            chat_template: Optional[str] = None

        mock_llm = MockLLM()
        mock_llm._tokenizer = mock_tokenizer

        task = Genstruct(
            name="genstruct",
            llm=mock_llm,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        inputs = task.format_input({"title": "Test Title", "content": "Test Content"})
        rendered = mock_tokenizer.apply_chat_template(inputs)

        assert "[[[Title]]] Test Title" in rendered
        assert "[[[Content]]] Test Content" in rendered
        assert "[[[User]]]" in rendered
