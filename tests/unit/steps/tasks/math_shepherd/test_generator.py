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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pytest

from distilabel.steps.tasks.math_shepherd.generator import (
    FEW_SHOTS_GSM8K,
    RULES_GSM8K,
    SYSTEM_PROMPT,
    MathShepherdGenerator,
)
from tests.unit.conftest import DummyLLM

if TYPE_CHECKING:
    from distilabel.typing import GenerateOutput


class MathShepherdGeneratorLLM(DummyLLM):
    M: Optional[int] = None
    structured: bool = False
    with_error: bool = False

    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "math-shepherd-generator"

    def generate(
        self, inputs: Dict[str, Any], num_generations: int = 1
    ) -> List["GenerateOutput"]:
        if self.structured:
            if self.M:
                solutions = [
                    {
                        "solution": "Step 1: Calculate the number of eggs Janet lays per day: 16 eggs/day. She eats 3 for breakfast and bakes 4 for muffins, so she uses a total of 3 + 4 = 7 eggs per day. Calculate the number of eggs she has left: 16 - 7 = <<16-7=9>>9 eggs/day. Step 2: Calculate the amount of money Janet makes from selling eggs at the farmers' market: 9 eggs/day * $2/egg = <<9*2=18>>$18/day. Step 3: Calculate the total number of books borrowed from Monday to Thursday: 40 * 4 = <<40*4=160>>160 books. Step 4: Calculate the total number of books borrowed in the entire week: 160 + 56 = <<160+56=216>>216 books The answer is: 18"
                    }
                    for _ in range(self.M)
                ]
            else:
                solutions = [
                    {
                        "solution": "Step 1: Calculate the number of eggs Janet lays per day: 16 eggs/day. She eats 3 for breakfast and bakes 4 for muffins, so she uses a total of 3 + 4 = 7 eggs per day. Calculate the number of eggs she has left: 16 - 7 = <<16-7=9>>9 eggs/day. Step 2: Calculate the amount of money Janet makes from selling eggs at the farmers' market: 9 eggs/day * $2/egg = <<9*2=18>>$18/day. Step 3: Calculate the total number of books borrowed from Monday to Thursday: 40 * 4 = <<40*4=160>>160 books. Step 4: Calculate the total number of books borrowed in the entire week: 160 + 56 = <<160+56=216>>216 books The answer is: 18"
                    }
                ]
            response = json.dumps({"solutions": solutions})
        else:
            response = """
Step 1: Calculate the number of books borrowed on a regular day (Monday to Thursday):
40 books per day

Step 2: Calculate the number of books borrowed on Friday, which is 40% higher than the daily average:
40 * 40/100 = <<40*40/100=16>>16 books
40 + 16 = <<40+16=56>>56 books

Step 3: Calculate the total number of books borrowed from Monday to Thursday:
40 * 4 = <<40*4=160>>160 books

Step 4: Calculate the total number of books borrowed in the entire week:
160 + 56 = <<160+56=216>>216 books

The answer is: 216 books."""
            if self.M:
                response = "---".join([response for _ in range(self.M)])
        return [
            {
                "generations": [response] * num_generations,
                "statistics": {
                    "input_tokens": [12] * num_generations,
                    "output_tokens": [12] * num_generations,
                },
            }
            for _ in range(len(inputs))
        ]


class TestMathShepherdGenerator:
    @pytest.mark.parametrize(
        "system_prompt",
        [
            None,  # Use the default
            SYSTEM_PROMPT,
        ],
    )
    @pytest.mark.parametrize(
        "extra_rules, few_shots",
        [
            (None, None),  # Use the default
            (RULES_GSM8K, FEW_SHOTS_GSM8K),
        ],
    )
    @pytest.mark.parametrize(
        "M",
        [None, 5],
    )
    def test_format_input(
        self,
        system_prompt: Optional[str],
        extra_rules: Optional[str],
        few_shots: Optional[str],
        M: Optional[int],
    ) -> None:
        task = MathShepherdGenerator(
            llm=MathShepherdGeneratorLLM(),
            system_prompt=system_prompt,
            extra_rules=extra_rules,
            few_shots=few_shots,
            M=M,
        )
        task.load()

        result = task.format_input(
            input={
                "instruction": "Krystian works in the library. He borrows an average of 40 books every day. Every Friday, his number of borrowed books is about 40% higher than the daily average. How many books does he borrow in a week if the library is open from Monday to Friday?"
            }
        )
        rendered_system_prompt = ""
        if system_prompt:
            rendered_system_prompt = result[0]["content"]
            if extra_rules:
                assert RULES_GSM8K in rendered_system_prompt
            if few_shots:
                assert FEW_SHOTS_GSM8K in rendered_system_prompt
            if M:
                assert (
                    "Generate 5 example solutions to the following problem,"
                    in result[1]["content"]
                )
        else:
            if M:
                assert (
                    "Generate 5 example solutions to the following problem,"
                    in result[0]["content"]
                )

    @pytest.mark.parametrize(
        "M, output_name",
        [
            (None, "golden_solution"),
            (3, "solutions"),
        ],
    )
    def test_process(self, M: Optional[int], output_name: str) -> None:
        task = MathShepherdGenerator(llm=MathShepherdGeneratorLLM(M=M), M=M)
        task.load()

        result = next(task.process([{"instruction": ""}]))[0][output_name]
        if M:
            assert len(result) == 3
            assert all(len(r) == 5 for r in result)
        else:
            assert len(result) == 5

    @pytest.mark.parametrize(
        "M, output_name",
        [
            # (None, "golden_solution"),  # golden solution doesn't need structured output
            (1, "solutions"),
            (3, "solutions"),
        ],
    )
    def test_process_structured_output(
        self, M: Optional[int], output_name: str
    ) -> None:
        task = MathShepherdGenerator(
            llm=MathShepherdGeneratorLLM(M=M, structured=True),
            use_default_structured_output=True,
            M=M,
        )
        task.load()

        result = next(task.process([{"instruction": ""}]))[0][output_name]
        if M:
            assert len(result) == M
            assert all(len(r) == 5 for r in result)
        else:
            assert len(result) == 5

    @pytest.mark.parametrize(
        "M, output_name, expected",
        [
            # (None, "golden_solution"),
            (
                2,
                "solutions",
                {
                    "instruction": "",
                    "solutions": [
                        [
                            "Step 1: Calculate the number of eggs Janet lays per day: 16 eggs/day. She eats 3 for breakfast and bakes 4 for muffins, so she uses a total of 3 + 4 = 7 eggs per day. Calculate the number of eggs she has left: 16 - 7 = <<16-7=9>>9 eggs/day.",
                            "Step 2: Calculate the amount of money Janet makes from selling eggs at the farmers' market: 9 eggs/day * $2/egg = <<9*2=18>>$18/day.",
                            "Step 3: Calculate the total number of books borrowed from Monday to Thursday: 40 * 4 = <<40*4=160>>160 books.",
                            "Step 4: Calculate the total number of books borrowed in the entire week: 160 + 56 = <<160+56=216>>216 books",
                            "The answer is: 18",
                        ],
                        [
                            "Step 1: Calculate the number of eggs Janet lays per day: 16 eggs/day. She eats 3 for breakfast and bakes 4 for muffins, so she uses a total of 3 + 4 = 7 eggs per day. Calculate the number of eggs she has left: 16 - 7 = <<16-7=9>>9 eggs/day.",
                            "Step 2: Calculate the amount of money Janet makes from selling eggs at the farmers' market: 9 eggs/day * $2/egg = <<9*2=18>>$18/day.",
                            "Step 3: Calculate the total number of books borrowed from Monday to Thursday: 40 * 4 = <<40*4=160>>160 books.",
                            "Step 4: Calculate the total number of books borrowed in the entire week: 160 + 56 = <<160+56=216>>216 books",
                            "The answer is: 18",
                        ],
                    ],
                },
            ),
            (
                3,
                "solutions",
                {
                    "instruction": "",
                    "solutions": [[], [], []],
                },
            ),
        ],
    )
    def test_format_output(self, M: int, output_name: str, expected) -> None:
        task = MathShepherdGenerator(
            llm=MathShepherdGeneratorLLM(M=M),
            M=M,
            use_default_structured_output=True,
        )
        task.load()

        # This is just to force an error in the generation so that the default is
        # returned and we can test it easily
        if M == 3:
            M += 1

        solutions = [
            {
                "solution": "Step 1: Calculate the number of eggs Janet lays per day: 16 eggs/day. She eats 3 for breakfast and bakes 4 for muffins, so she uses a total of 3 + 4 = 7 eggs per day. Calculate the number of eggs she has left: 16 - 7 = <<16-7=9>>9 eggs/day. Step 2: Calculate the amount of money Janet makes from selling eggs at the farmers' market: 9 eggs/day * $2/egg = <<9*2=18>>$18/day. Step 3: Calculate the total number of books borrowed from Monday to Thursday: 40 * 4 = <<40*4=160>>160 books. Step 4: Calculate the total number of books borrowed in the entire week: 160 + 56 = <<160+56=216>>216 books The answer is: 18"
            }
            for _ in range(M)
        ]
        solutions = json.dumps({"solutions": solutions})
        result = task.format_output(solutions, {"instruction": ""})
        assert result == expected
