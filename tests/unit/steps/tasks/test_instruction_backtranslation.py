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

from typing import Any, List

from distilabel.llms.base import LLM
from distilabel.llms.typing import GenerateOutput
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.instruction_backtranslation import (
    InstructionBacktranslation,
)
from distilabel.steps.tasks.typing import ChatType


class InstructionBacktranslationLLM(LLM):
    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "instruction-backtranslation-model"

    def generate(
        self, inputs: List[ChatType], num_generations: int = 1, **kwargs: Any
    ) -> List[GenerateOutput]:
        return [
            ["This is the reason. Score: 1" for _ in range(num_generations)]
            for _ in inputs
        ]


class TestInstructionBacktranslation:
    def test_format_input(self) -> None:
        task = InstructionBacktranslation(
            name="instruction-backtranslation",
            llm=InstructionBacktranslationLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert task.format_input(
            {"instruction": "instruction", "generation": "generation"}
        ) == [
            {
                "role": "user",
                "content": 'Below is an instruction from an user and a candidate answer. Evaluate whether or not the answer is a good example of how AI Assistant should respond to the user’s instruction. Please assign a score using the following 5-point scale:\n1: It means the answer is incomplete, vague, off-topic, controversial, or not exactly what the user asked for. For example, some content seems missing, numbered list does not start from the beginning, the opening sentence repeats user’s question. Or the response is from another person’s perspective with their personal experience (e.g. taken from blog posts), or looks like an answer from a forum. Or it contains promotional text, navigation text, or other irrelevant information.\n2: It means the answer addresses most of the asks from the user. It does not directly address the user’s question. For example, it only provides a high-level methodology instead of the exact solution to user’s question.\n3: It means the answer is helpful but not written by an AI Assistant. It addresses all the basic asks from the user. It is complete and self contained with the drawback that the response is not written from an AI assistant’s perspective, but from other people’s perspective. The content looks like an excerpt from a blog post, web page, or web search results. For example, it contains personal experience or opinion, mentions comments section, or share on social media, etc.\n4: It means the answer is written from an AI assistant’s perspective with a clear focus of addressing the instruction. It provide a complete, clear, and comprehensive response to user’s question or instruction without missing or irrelevant information. It is well organized, self-contained, and written in a helpful tone. It has minor room for improvement, e.g. more concise and focused.\n5: It means it is a perfect answer from an AI Assistant. It has a clear focus on being a helpful AI Assistant, where the response looks like intentionally written to address the user’s question or instruction without any irrelevant sentences. The answer provides high quality content, demonstrating expert knowledge in the area, is very well written, logical, easy-to-follow, engaging and insightful.\nPlease first provide a brief reasoning you used to derive the rating score, and then write "Score: <rating>" in the last line.\n\ninstruction\ngeneration\n',
            }
        ]

    def test_format_output(self) -> None:
        task = InstructionBacktranslation(
            name="instruction-backtranslation",
            llm=InstructionBacktranslationLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert task.format_output("This is the reason. Score: 1", {}) == {
            "score": 1,
            "reason": "This is the reason.",
        }

    def test_process(self) -> None:
        task = InstructionBacktranslation(
            name="instruction-backtranslation",
            llm=InstructionBacktranslationLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
            add_raw_input=False,
        )
        task.load()

        assert next(task.process([{"instruction": "test", "generation": "A"}])) == [
            {
                "instruction": "test",
                "generation": "A",
                "score": 1,
                "reason": "This is the reason.",
                "model_name": "instruction-backtranslation-model",
                "distilabel_metadata": {
                    "raw_output_instruction-backtranslation": "This is the reason. Score: 1"
                },
            }
        ]
