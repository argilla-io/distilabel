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

import pytest

from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.text_generation import ChatGeneration, TextGeneration
from tests.unit.conftest import DummyLLM


class TestTextGeneration:
    def test_format_input(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyLLM()
        task = TextGeneration(
            name="task", llm=llm, pipeline=pipeline, use_system_prompt=False
        )

        assert task.format_input({"instruction": "test", "system_prompt": "test"}) == [
            {"role": "user", "content": "test"}
        ]

    def test_format_input_with_system_prompt(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyLLM()
        task = TextGeneration(
            name="task",
            llm=llm,
            pipeline=pipeline,
            use_system_prompt=True,
        )

        assert task.format_input({"instruction": "test", "system_prompt": "test"}) == [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ]

    def test_format_input_errors(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyLLM()
        task = TextGeneration(
            name="task", llm=llm, pipeline=pipeline, use_system_prompt=True
        )

        with pytest.raises(
            ValueError,
            match=r"Providing \`instruction\` formatted as an OpenAI chat / conversation is deprecated",
        ):
            task.format_input({"instruction": [{"role": "user", "content": "test"}]})

        with pytest.raises(
            ValueError, match=r"Input \`instruction\` must be a string. Got: 1."
        ):
            task.format_input({"instruction": 1})

        with pytest.warns(
            UserWarning,
            match=r"\`use_system_prompt\` is set to \`True\`, but no \`system_prompt\` in input batch, so it will be ignored.",
        ):
            assert task.format_input({"instruction": "test"}) == [
                {"role": "user", "content": "test"}
            ]

    def test_process(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyLLM()
        task = TextGeneration(name="task", llm=llm, pipeline=pipeline)

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


class TestChatGeneration:
    def test_format_input(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        llm = DummyLLM()
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
        llm = DummyLLM()
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
        llm = DummyLLM()
        task = ChatGeneration(name="task", llm=llm, pipeline=pipeline)

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
