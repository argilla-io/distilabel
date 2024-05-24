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

from distilabel.pipeline.local import Pipeline
from distilabel.steps.formatting.conversation import ConversationTemplate


class TestConversationTemplate:
    def test_process(self) -> None:
        conversation_template = ConversationTemplate(
            name="conversation_template",
            pipeline=Pipeline(name="unit-test"),
        )

        result = next(
            conversation_template.process([{"instruction": "Hello", "response": "Hi"}])
        )

        assert result == [
            {
                "instruction": "Hello",
                "response": "Hi",
                "conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            }
        ]
