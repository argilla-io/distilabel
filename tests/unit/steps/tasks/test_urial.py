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

from distilabel.steps.tasks.urial import URIAL
from tests.unit.conftest import DummyLLM


class TestURIAL:
    def test_format_input(self) -> None:
        task = URIAL(llm=DummyLLM())
        task.load()
        assert task.format_input({"instruction": "test"}) == [
            {
                "role": "user",
                "content": '# Instruction\n\nBelow is a list of conversations between a human and an AI assistant (you). \nUsers place their queries under "# User:", and your responses are under  "# Assistant:".\nYou are a helpful, respectful, and honest assistant.\nYou should always answer as helpfully as possible while ensuring safety.\nYour answers should be well-structured and provide detailed information. They should also have an engaging tone.\nYour responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\nYour response must be socially responsible, and thus you can refuse to answer some controversial topics.\n\n\n# User:\n\ntest\n\n# Assistant:',
            }
        ]

    def test_format_input_with_conversation(self) -> None:
        task = URIAL(llm=DummyLLM())
        task.load()
        assert task.format_input(
            {
                "conversation": [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "test"},
                    {"role": "user", "content": "test"},
                ]
            }
        ) == [
            {
                "role": "user",
                "content": '# Instruction\n\nBelow is a list of conversations between a human and an AI assistant (you). \nUsers place their queries under "# User:", and your responses are under  "# Assistant:".\nYou are a helpful, respectful, and honest assistant.\nYou should always answer as helpfully as possible while ensuring safety.\nYour answers should be well-structured and provide detailed information. They should also have an engaging tone.\nYour responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\nYour response must be socially responsible, and thus you can refuse to answer some controversial topics.\n\n\n# User:\n\ntest\n\n# Assistant:\n\ntest\n\n# User:\n\ntest\n\n# Assistant:',
            }
        ]

    def test_format_input_raise_valueerror(self) -> None:
        task = URIAL(llm=DummyLLM())
        task.load()

        with pytest.raises(ValueError, match="The last message must be from the user."):
            assert task.format_input(
                {
                    "conversation": [
                        {"role": "user", "content": "test"},
                        {"role": "assistant", "content": "test"},
                    ]
                }
            )

    def test_format_output(self) -> None:
        task = URIAL(llm=DummyLLM())
        task.load()

        assert task.format_output(
            output=" \n\noutput\n\n# User:", input={"instruction": "test"}
        ) == {
            "generation": "output",
        }
