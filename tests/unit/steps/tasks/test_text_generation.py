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

from distilabel.errors import DistilabelUserError
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.text_generation import ChatGeneration, TextGeneration
from tests.unit.conftest import DummyAsyncLLM


class TestTextGeneration:
    def test_format_input(self) -> None:
        llm = DummyAsyncLLM()
        task = TextGeneration(name="task", llm=llm)
        task.load()

        assert task.format_input({"instruction": "test"}) == [
            {"role": "user", "content": "test"}
        ]

    def test_format_input_with_system_prompt(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = TextGeneration(
            name="task", llm=llm, pipeline=pipeline, system_prompt="test"
        )
        task.load()

        assert task.format_input({"instruction": "test"}) == [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ]

    def test_format_input_with_row_system_prompt(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = TextGeneration(name="task", llm=llm, pipeline=pipeline)
        task.load()

        assert task.format_input({"instruction": "test", "system_prompt": "test"}) == [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ]

    def test_format_input_with_row_system_prompt_and_system_prompt(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = TextGeneration(
            name="task", llm=llm, pipeline=pipeline, system_prompt="i won't be used"
        )
        task.load()

        assert task.format_input({"instruction": "test", "system_prompt": "test"}) == [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ]

    def test_format_input_errors(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = TextGeneration(
            name="task", llm=llm, pipeline=pipeline, use_system_prompt=True
        )
        task.load()

        with pytest.raises(
            ValueError,
            match=r"Providing \`instruction\` formatted as an OpenAI chat / conversation is deprecated",
        ):
            task.format_input({"instruction": [{"role": "user", "content": "test"}]})

        with pytest.raises(
            ValueError, match=r"Input \`instruction\` must be a string. Got: 1."
        ):
            task.format_input({"instruction": 1})

    def test_process(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = TextGeneration(
            name="task", llm=llm, pipeline=pipeline, add_raw_input=False
        )
        task.load()

        assert next(task.process([{"instruction": "test"}])) == [
            {
                "instruction": "test",
                "generation": "output",
                "model_name": "test",
                "distilabel_metadata": {
                    "raw_output_task": "output",
                },
            }
        ]

    @pytest.mark.parametrize(
        "template, columns, sample",
        [
            (None, "instruction", {"instruction": "INSTRUCTION"}),
            (
                "Document:\n{{ document }}\n\nQuestion: {{ question }}\n\nPlease provide a clear and concise answer to the question based on the information in the document and your general knowledge:",
                ["document", "question"],
                {"document": "DOCUMENT", "question": "QUESTION"},
            ),
            (
                "Generate a clear, single-sentence instruction based on the following examples:\n\n{% for example in examples %}\nExample {{ loop.index }}:\nInstruction: {{ example }}\n\n{% endfor %}\nNow, generate a new instruction in a similar style:\n",
                "examples",
                {"examples": ["example1", "example2"]},
            ),
        ],
    )
    def test_format_input_custom_columns(
        self,
        template: str,
        columns: Union[str, List[str]],
        sample: Dict[str, Any],
    ) -> None:
        task = TextGeneration(
            llm=DummyAsyncLLM(),
            system_prompt=None,
            template=template,
            columns=columns,
            add_raw_input=False,
            add_raw_output=False,
        )
        task.load()

        # Check the input from the sample are present in the formatted input
        result = task.format_input(sample)[0]["content"]
        values = list(sample.values())

        if isinstance(values[0], list):
            values = values[0]
        assert all(v in result for v in values)

    @pytest.mark.parametrize(
        "template, columns, sample",
        [
            (
                "This is a {{ custom }} template",
                "instruction",
                {"other": "INSTRUCTION"},
            ),
        ],
    )
    def test_format_input_custom_columns_expected_errors(
        self,
        template: str,
        columns: Union[str, List[str]],
        sample: Dict[str, Any],
    ) -> None:
        task = TextGeneration(
            llm=DummyAsyncLLM(),
            system_prompt=None,
            template=template,
            columns=columns,
            add_raw_input=False,
            add_raw_output=False,
        )
        with pytest.raises(DistilabelUserError):
            task.load()


class TestChatGeneration:
    def test_format_input(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = ChatGeneration(name="task", llm=llm, pipeline=pipeline)

        assert task.format_input(
            {
                "messages": [
                    {"role": "system", "content": "You're a helpful assistant"},
                    {"role": "user", "content": "How much is 2+2?"},
                ]
            }
        ) == [
            {"role": "system", "content": "You're a helpful assistant"},
            {"role": "user", "content": "How much is 2+2?"},
        ]

    def test_format_input_errors(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = ChatGeneration(name="task", llm=llm, pipeline=pipeline)

        with pytest.raises(ValueError, match="The last message must be from the user"):
            task.format_input(
                {
                    "messages": [
                        {"role": "user", "content": "How much is 2+2?"},
                        {"role": "assistant", "content": "4"},
                    ]
                }
            )

    def test_process(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyAsyncLLM()
        task = ChatGeneration(
            name="task", llm=llm, pipeline=pipeline, add_raw_input=False
        )

        assert next(
            task.process(
                [
                    {
                        "messages": [
                            {"role": "user", "content": "Tell me a joke."},
                        ]
                    }
                ]
            )
        ) == [
            {
                "messages": [{"role": "user", "content": "Tell me a joke."}],
                "generation": "output",
                "model_name": "test",
                "distilabel_metadata": {"raw_output_task": "output"},
            }
        ]
