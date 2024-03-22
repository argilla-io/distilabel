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

from typing import Any, Dict, Union

import pytest
from distilabel.pipeline.local import Pipeline
from distilabel.steps.task.ultracm import UltraCM

from tests.unit.steps.task.utils import DummyLLM


class TestUltraCM:
    def test_format_input(self) -> None:
        task = UltraCM(name="ultracm", llm=DummyLLM(), pipeline=Pipeline())
        task.load()

        result = task.format_input(
            input={
                "instruction": "instruction 1",
                "completion": "answer 1",
            }
        )

        assert result == [
            {
                "role": "system",
                "content": "User: A one-turn chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.</s>",
            },
            {
                "role": "user",
                "content": "User: Given my answer to an instruction, your role is to provide specific and constructive feedback for me. You should find the best way for me to learn from your feedback and improve my performance.\n\nYou should consider multiple aspects of my answer, including helpfulness, truthfulness, honesty, and to what extent the answer follows instructions.\n---\n\n### Instruction\ninstruction 1\n\n### Answer\nanswer 1\n---\n\nPlease act as a teacher and provide specific and constructive feedback. Besides describing the weaknesses of the answer, you should also provide specific suggestions to guide me toward understanding how to improve. Please note, however, that your suggestions should help me better complete the instructions, but you should not introduce new requirements that are not mentioned in the instructions. Your feedback should focus on enhancing my ability to think critically and respond accurately. However, never explicitly provide the reference answer, nor do polite phrases be required. Only respond with concise feedback in chat style. Finally, score the overall quality of the answer from 1 to 10, where 1 is the worst and 10 is the best.\n\n*Format*\n### Feedback\nOverall Score: [1-10]\n[Your feedback]\n\n---</s>\nAssistant: ### Feedback\nOverall Score: ",
            },
        ]

    @pytest.mark.parametrize(
        "output, expected",
        [
            (
                "1 \nThis is a critique.",
                {"score": 1.0, "critique": "This is a critique.", "raw_output": None},
            ),
            (
                None,
                {"score": None, "critique": None, "raw_output": None},
            ),
            (
                "Text that cannot be parsed.",
                {
                    "score": None,
                    "critique": None,
                    "raw_output": "Text that cannot be parsed.",
                },
            ),
        ],
    )
    def test_format_output(
        self, output: Union[str, None], expected: Dict[str, Any]
    ) -> None:
        task = UltraCM(name="ultracm", llm=DummyLLM(), pipeline=Pipeline())
        task.load()

        result = task.format_output(output=output)
        import json

        print(json.dumps(result, indent=2))

        assert result == expected
