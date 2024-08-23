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

from distilabel.steps.reward_model import RewardModelScore


class TestRewardModelScore:
    def test_process(self) -> None:
        step = RewardModelScore(
            model="OpenAssistant/reward-model-deberta-v3-large-v2",
        )

        step.load()

        step._tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"

        result = next(
            step.process(
                inputs=[
                    {
                        "instruction": "How much is 2+2?",
                        "response": "The output of 2+2 is 4",
                    },
                    {"instruction": "How much is 2+2?", "response": "4"},
                ]
            )
        )

        assert result == [
            {
                "instruction": "How much is 2+2?",
                "response": "The output of 2+2 is 4",
                "score": pytest.approx(0.28374260663986206, abs=1e-5),
            },
            {
                "instruction": "How much is 2+2?",
                "response": "4",
                "score": pytest.approx(-4.194192409515381, abs=1e-5),
            },
        ]

    def test_process_with_conversation(self) -> None:
        step = RewardModelScore(
            model="OpenAssistant/reward-model-deberta-v3-large-v2",
        )

        step.load()

        step._tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"

        result = next(
            step.process(
                inputs=[
                    {
                        "conversation": [
                            {"role": "user", "content": "How much is 2+2?"},
                            {"role": "assistant", "content": "The output of 2+2 is 4"},
                        ],
                    },
                    {
                        "conversation": [
                            {"role": "user", "content": "How much is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ],
                    },
                ]
            )
        )

        assert result == [
            {
                "conversation": [
                    {"role": "user", "content": "How much is 2+2?"},
                    {"role": "assistant", "content": "The output of 2+2 is 4"},
                ],
                "score": pytest.approx(0.28374260663986206, abs=1e-5),
            },
            {
                "conversation": [
                    {"role": "user", "content": "How much is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                "score": pytest.approx(-4.194192409515381, abs=1e-5),
            },
        ]
