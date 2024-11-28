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

import textwrap
from typing import TYPE_CHECKING, Any, Dict, List

import pytest

from distilabel.steps.tasks.math_shepherd.completer import MathShepherdCompleter
from tests.unit.conftest import DummyLLM

if TYPE_CHECKING:
    from distilabel.models.llms.typing import GenerateOutput


class MathShepherdCompleterLLM(DummyLLM):
    N: int = 3

    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "math-shepherd-completer"

    def generate(
        self, inputs: Dict[str, Any], num_generations: int = 1
    ) -> List["GenerateOutput"]:
        if self.N == 1:
            response = textwrap.dedent("""
                Step 1: Determine the total number of eggs Janet collects per day: Janet's ducks lay 16 eggs per day.
                Step 2: Calculate the number of eggs Janet uses for herself per day: She eats three for breakfast and bakes muffins with four eggs, for a total of 3 + 4 = <<3+4=7>>7 eggs.
                Step 3: Calculate the number of eggs Janet has left to sell per day: 16 - 7 = <<16-7=9>>9 eggs.
                Step 4: Calculate the total amount Janet makes at the farmers' market per day: 9 * $2 = <<9*2=18>>18 dollars.

                The answer is: $18""")
        else:
            response = textwrap.dedent("""
                Step 2: Janet's ducks lay 16 eggs per day, and she uses 7 for eating and baking. So the number of eggs she has left is 16 - 7 = <<16-7=9>>9.
                Step 3: Janet sells the remaining 9 eggs for $2 each, so she makes 9 * 2 = <<9*2=18>>18 dollars every day at the farmers' market.
                The answer is: 18

                ---

                Step 2: Janet's ducks lay 16 eggs per day, and she uses 3 for eating and bakes 4 for her friends, so she has 16 - 7 = <<16-7=9>>9 eggs left.
                Step 3: Selling the 9 eggs at $2 each, she makes 9 * 2 = <<9*2=18>>18 dollars every day.
                The answer is: 18

                ---

                Step 2: Janets ducks lay 16 eggs per day. She eats 3 and bakes 4, so she has 16 - (3 + 4) = 16 - 7 = 9 eggs left.
                Step 3: She sells the 9 eggs for $2 each, which means she makes 9 * $2 = $<<9*2=18>>18.
                The answer is: 18""")
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


DUMMY_STEPS = [
    "Step 1: Determine the total number of eggs Janet collects per day: Janet's ducks lay 16 eggs per day.",
    "Step 2: Calculate the number of eggs Janet uses for herself per day: She eats three for breakfast and bakes muffins with four eggs, for a total of 3 + 4 = <<3+4=7>>7 eggs.",
    "Step 3: Calculate the number of eggs Janet has left to sell per day: 16 - 7 = <<16-7=9>>9 eggs.",
    "Step 4: Calculate the total amount Janet makes at the farmers' market per day: 9 * $2 = <<9*2=18>>18 dollars.",
    "The answer is: $18",
]


class TestMathShepherdCompleter:
    @pytest.mark.parametrize(
        "steps, num_completions",
        [
            (DUMMY_STEPS, 3),
            # This would be the same case as having the problem already solved in a single step,
            # there's nothing else we have to do
            (DUMMY_STEPS[-2:], 0),
        ],
    )
    def test_prepare_completions(self, steps: List[str], num_completions: int) -> None:
        task = MathShepherdCompleter(llm=MathShepherdCompleterLLM(N=1), N=1)
        task.load()
        instruction = "Krystian works in the library. He borrows an average of 40 books every day. Every Friday, his number of borrowed books is about 40% higher than the daily average. How many books does he borrow in a week if the library is open from Monday to Friday?"
        prepared_inputs = task._prepare_completions(instruction, steps)
        assert len(prepared_inputs) == num_completions

    def test_process(self) -> None:
        task = MathShepherdCompleter(
            llm=MathShepherdCompleterLLM(N=3),
            N=3,
        )
        task.load()
        result = next(
            task.process(
                [
                    {
                        "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                        "golden_solution": [
                            "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
                            "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.",
                            "The answer is: 18",
                        ],
                        "solutions": [
                            [
                                "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
                                "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.",
                                "The answer is: 18",
                            ],
                            [
                                "Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking.",
                                "Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day.",
                                "Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18.",
                                "The answer is: 18",
                            ],
                        ],
                    },
                ]
            )
        )

        assert result == [
            {
                "golden_solution": [
                    "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
                    "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.",
                    "The answer is: 18",
                ],
                "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "model_name": "math-shepherd-completer",
                "solutions": [
                    [
                        "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. +",
                        "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market. The answer is: 18 +",
                    ],
                    [
                        "Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking. +",
                        "Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day. +",
                        "Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18. The answer is: 18 +",
                    ],
                ],
                "distilabel_metadata": {
                    "statistics_math_shepherd_completer_0": {
                        "input_tokens": [12],
                        "output_tokens": [12],
                    },
                    "raw_input_math_shepherd_completer_0": [
                        {
                            "role": "system",
                            "content": 'You are a math teacher who helps students by breaking down word problems into clear, logical steps.\nWhen given a problem statement and any number of initial step, generate the remaining steps needed to reach the final answer.\nEach step should:\n\n- Build logically on previous steps\n- Explain the reasoning in natural language\n- Lead to the final answer\n- Multiple solution paths are acceptable\n- Steps should be concise but clear\n- Each calculation should be shown explicitly\n- The final answer must be clearly stated\n- The number of steps may vary based on the solution approach\n\n# Format requirements:\n- Each step should be numbered sequentially, continuing from the last given step\n- The final step should clearly state "The answer is: [result]"\n- Each step can use different approaches but must be mathematically valid\n\n# Rules:\n- All calculations must be shown within <<>> brackets\n- Basic operations: use * for multiplication, / for division, + for addition, - for subtraction\n- Write the full calculation and result, e.g., <<5*10=50>>50\n\n# Examples:\n## Input\nKrystian works in the library. He borrows an average of 40 books every day. Every Friday, his number of borrowed books is about 40% higher than the daily average. How many books does he borrow in a week if the library is open from Monday to Friday?\nStep 1: On Friday, Krystian borrows 40 * 0.4 = <<40*0.4=16>>16 more books than on a regular day.\n\n## Output 1\nStep 2: On Friday, Krystian borrows 40 + 16 = <<40+16=56>>56 books in total.\nStep 3: For the other 4 days (Monday to Thursday), he borrows 40 * 4 = <<40*4=160>>160 books.\nStep 4: The total books for the week is 160 + 56 = <<160+56=216>>216. The answer is: 216\n\n## Output 2\nStep 2: In total, he borrows 40 + 16 = <<40+16=56>>56 books on Friday.\nStep 3: For the whole week (4 regular days plus Friday), the total is (40 * 4) + 56 = <<(40*4)+56=216>>216. The answer is: 216\n\n## Output 3\nStep 2: On Friday, he borrows 40 + 40/100 * 40 = <<40+40/100*40=56>>56 books.\nStep 3: In a week, he borrows 5.7 * 7 = <<5.7*7=40>>40 books. The answer is: 40',
                        },
                        {
                            "role": "user",
                            "content": "Generate 3 example solutions to the same problem, separated by a single `---` and nothing else.\nResponse format:\n```\nStep i: step i explanation.\nStep i+1: step i+1 explanation.\nThe answer is: X\n\n---\n\nStep 2: step i explanation.\nStep 3: step i+1 explanation.\nThe answer is: Y\n```\n\nThis is the problem:\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
                        },
                    ],
                    "raw_output_math_shepherd_completer_0": """\nStep 2: Janet's ducks lay 16 eggs per day, and she uses 7 for eating and baking. So the number of eggs she has left is 16 - 7 = <<16-7=9>>9.
Step 3: Janet sells the remaining 9 eggs for $2 each, so she makes 9 * 2 = <<9*2=18>>18 dollars every day at the farmers' market.
The answer is: 18

---

Step 2: Janet's ducks lay 16 eggs per day, and she uses 3 for eating and bakes 4 for her friends, so she has 16 - 7 = <<16-7=9>>9 eggs left.
Step 3: Selling the 9 eggs at $2 each, she makes 9 * 2 = <<9*2=18>>18 dollars every day.
The answer is: 18

---

Step 2: Janets ducks lay 16 eggs per day. She eats 3 and bakes 4, so she has 16 - (3 + 4) = 16 - 7 = 9 eggs left.
Step 3: She sells the 9 eggs for $2 each, which means she makes 9 * $2 = $<<9*2=18>>18.
The answer is: 18""",
                },
            }
        ]

    def test_auto_label(self):
        inputs = [
            {
                "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "golden_solution": [
                    "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
                    "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.",
                    "The answer is: 18",
                ],
                "solutions": [
                    [
                        "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
                        "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.",
                        "The answer is: 18",
                    ],
                    [
                        "Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking.",
                        "Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day.",
                        "Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18.",
                        "The answer is: 18",
                    ],
                ],
            },
        ]
        N = 3
        task = MathShepherdCompleter(
            llm=MathShepherdCompleterLLM(N=N),
            N=N,
            add_raw_input=False,
            add_raw_output=False,
        )
        task.load()
        final_outputs = [
            [
                [
                    "Step 2: Janet sells 9 duck eggs at the farmers' market, so she makes 9 * $1 = $<<9*1=9>>9 from selling the eggs.",
                    "The answer is: $9",
                ],
                [
                    "Step 1: Janet lays 16 eggs per day, eats 3 for breakfast, uses 4 for baking, so she has 16 - 3 - 4 = 9 eggs left.",
                    "Step 2: Since Janet sells 9 eggs a day, and each egg is sold for $1, she makes 9 * $1 = $<<9*1=9>>9.",
                    "The answer is: $9",
                ],
                [
                    "Step 1: Janet lays 16 eggs per day, eats 3, uses 4 for baking which leaves her with 16 - 3 - 4 = 9 eggs.",
                    "Step 2: Since she sells the eggs for $1 each, she makes 9 * $1 = $<<9*1=9>>9.",
                    "The answer is: $9",
                ],
            ],
            [
                [
                    "Step 3: To determine how many eggs Jan's sells at the market, we need to subtract the eggs she uses (7) from the total number of eggs laid (16), which is 16 - 7 = <<16-7=9>>9.",
                    "Step 4: Since she sells 9 eggs for $2 each, we multiply 9 * 2 = <<9*2=18>>18 to find out her daily earnings.",
                    "The answer is: 18",
                ],
                [
                    "Step 2: Jan's ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking.",
                    "Step 3: To find the number of eggs Jan's sells at the market, we subtract the eggs she uses (7) from the total number of eggs laid (16), which is 16 - 7 = <<16-7=9>>9.",
                    "Step 4: Since she sells 9 eggs for $2 each, we multiply 9 * 2 = <<9*2=18>>18 to find out her daily earnings.",
                    "The answer is: 18",
                ],
                [
                    "Step 2: Jan's ducks lay 16 eggs per day, and she uses 7 for eating and baking.",
                    "Step 3: To find the number of eggs Jan's sells at the market, we calculate 16 - 7 = <<16-7=9>>9.",
                    "Step 4: Since she sells 9 eggs for $2 each, we multiply 9 * 2 = <<9*2=18>>18 to find out her daily earnings.",
                    "The answer is: 18",
                ],
            ],
            [
                [
                    "Step 1: Janet's ducks lay 16 eggs per day. She eats 3 eggs and bakes 4 eggs.",
                    "Step 2: So, she uses 3 + 4 = <<3+4=7>>7 eggs for eating and baking.",
                    "Step 3: She sells the remaining eggs, which is 16 - 7 = <<16-7=9>>9 duck eggs every day.",
                    "Step 4: She sells each egg for $2, so the total amount she makes is 9 * 2 = <<9*2=18>>18 dollars every day.",
                    "The answer is: 18",
                ],
                [
                    "Step 1: Janet's ducks lay 16 eggs per day.",
                    "Step 2: She eats 3 eggs and bakes 4 eggs, which is a total of 3 + 4 = <<3+4=7>>7 eggs.",
                    "Step 3: She sells the remaining eggs, which is 16 - 7 = <<16-7=9>>9 duck eggs every day.",
                    "Step 4: Since she sells each egg for $2, she makes 9 * 2 = <<9*2=18>>18 dollars every day.",
                    "The answer is: 18",
                ],
                [
                    "Step 1: Janet's ducks lay 16 eggs per day.",
                    "Step 2: She consumes 7 eggs for eating and baking, which means she has 16 - 7 = <<16-7=9>>9 eggs left.",
                    "Step 3: She sells each egg for $2, so she makes 9 * 2 = <<9*2=18>>18 dollars every day.",
                    "The answer is: 18",
                ],
            ],
        ]

        golden_answers = ["The answer is: 18", "The answer is: 18"]
        input_positions = [(0, 0, 0), (0, 1, 0), (0, 1, 1)]
        statistics = [
            {"input_tokens": [12], "output_tokens": [12]},
            {"input_tokens": [12], "output_tokens": [12]},
            {"input_tokens": [12], "output_tokens": [12]},
        ]
        raw_outputs = [
            [
                "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. -",
                "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market. The answer is: 18 +",
            ],
            [
                "Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking. +",
                "Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day. +",
                "Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18. The answer is: 18 +",
            ],
        ]
        raw_inputs = [
            {
                "role": "system",
                "content": 'You are a math teacher who helps students by breaking down word problems into clear, logical steps.\nWhen given a problem statement and any number of initial step, generate the remaining steps needed to reach the final answer.\nEach step should:\n\n- Build logically on previous steps\n- Explain the reasoning in natural language\n- Lead to the final answer\n- Multiple solution paths are acceptable\n- Steps should be concise but clear\n- Each calculation should be shown explicitly\n- The final answer must be clearly stated\n- The number of steps may vary based on the solution approach\n\n# Format requirements:\n- Each step should be numbered sequentially, continuing from the last given step\n- The final step should clearly state "The answer is: [result]"\n- Each step can use different approaches but must be mathematically valid\n\n# Rules:\n- All calculations must be shown within <<>> brackets\n- Basic operations: use * for multiplication, / for division, + for addition, - for subtraction\n- Write the full calculation and result, e.g., <<5*10=50>>50\n\n# Examples:\n## Input\nKrystian works in the library. He borrows an average of 40 books every day. Every Friday, his number of borrowed books is about 40% higher than the daily average. How many books does he borrow in a week if the library is open from Monday to Friday?\nStep 1: On Friday, Krystian borrows 40 * 0.4 = <<40*0.4=16>>16 more books than on a regular day.\n\n## Output 1\nStep 2: On Friday, Krystian borrows 40 + 16 = <<40+16=56>>56 books in total.\nStep 3: For the other 4 days (Monday to Thursday), he borrows 40 * 4 = <<40*4=160>>160 books.\nStep 4: The total books for the week is 160 + 56 = <<160+56=216>>216. The answer is: 216\n\n## Output 2\nStep 2: In total, he borrows 40 + 16 = <<40+16=56>>56 books on Friday.\nStep 3: For the whole week (4 regular days plus Friday), the total is (40 * 4) + 56 = <<(40*4)+56=216>>216. The answer is: 216\n\n## Output 3\nStep 2: On Friday, he borrows 40 + 40/100 * 40 = <<40+40/100*40=56>>56 books.\nStep 3: In a week, he borrows 5.7 * 7 = <<5.7*7=40>>40 books. The answer is: 40',
            },
            {
                "role": "user",
                "content": "Generate 3 example solutions to the same problem, separated by a single `---` and nothing else.\nResponse format:\n```\nStep i: step i explanation.\nStep i+1: step i+1 explanation.\nThe answer is: X\n\n---\n\nStep 2: step i explanation.\nStep 3: step i+1 explanation.\nThe answer is: Y\n```\n\nThis is the problem:\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
            },
        ]
        results = task._auto_label(
            inputs,
            final_outputs,
            input_positions,
            golden_answers,
            statistics,
            raw_outputs,
            raw_inputs,
        )
        assert results == [
            {
                "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "golden_solution": [
                    "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
                    "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.",
                    "The answer is: 18",
                ],
                "solutions": [
                    [
                        "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. -",
                        "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market. The answer is: 18 +",
                    ],
                    [
                        "Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking. +",
                        "Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day. +",
                        "Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18. The answer is: 18 +",
                    ],
                ],
                "model_name": "math-shepherd-completer",
                "distilabel_metadata": {
                    "statistics_math_shepherd_completer_0": {
                        "input_tokens": [12],
                        "output_tokens": [12],
                    }
                },
            }
        ]

    def test_auto_label_with_errors(self):
        inputs = [
            {
                "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "golden_solution": [
                    "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
                    "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.",
                    "The answer is: 18",
                ],
                "solutions": [[], [], []],
            },
        ]
        N = 3
        task = MathShepherdCompleter(
            llm=MathShepherdCompleterLLM(N=N),
            N=N,
            add_raw_input=False,
            add_raw_output=False,
        )
        task.load()
        final_outputs = [
            [
                [
                    "Step 2: Janet sells 9 duck eggs at the farmers' market, so she makes 9 * $1 = $<<9*1=9>>9 from selling the eggs.",
                    "The answer is: $9",
                ],
                [
                    "Step 1: Janet lays 16 eggs per day, eats 3 for breakfast, uses 4 for baking, so she has 16 - 3 - 4 = 9 eggs left.",
                    "Step 2: Since Janet sells 9 eggs a day, and each egg is sold for $1, she makes 9 * $1 = $<<9*1=9>>9.",
                    "The answer is: $9",
                ],
                [
                    "Step 1: Janet lays 16 eggs per day, eats 3, uses 4 for baking which leaves her with 16 - 3 - 4 = 9 eggs.",
                    "Step 2: Since she sells the eggs for $1 each, she makes 9 * $1 = $<<9*1=9>>9.",
                    "The answer is: $9",
                ],
            ],
            [
                [
                    "Step 3: To determine how many eggs Jan's sells at the market, we need to subtract the eggs she uses (7) from the total number of eggs laid (16), which is 16 - 7 = <<16-7=9>>9.",
                    "Step 4: Since she sells 9 eggs for $2 each, we multiply 9 * 2 = <<9*2=18>>18 to find out her daily earnings.",
                    "The answer is: 18",
                ],
                [
                    "Step 2: Jan's ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking.",
                    "Step 3: To find the number of eggs Jan's sells at the market, we subtract the eggs she uses (7) from the total number of eggs laid (16), which is 16 - 7 = <<16-7=9>>9.",
                    "Step 4: Since she sells 9 eggs for $2 each, we multiply 9 * 2 = <<9*2=18>>18 to find out her daily earnings.",
                    "The answer is: 18",
                ],
                [
                    "Step 2: Jan's ducks lay 16 eggs per day, and she uses 7 for eating and baking.",
                    "Step 3: To find the number of eggs Jan's sells at the market, we calculate 16 - 7 = <<16-7=9>>9.",
                    "Step 4: Since she sells 9 eggs for $2 each, we multiply 9 * 2 = <<9*2=18>>18 to find out her daily earnings.",
                    "The answer is: 18",
                ],
            ],
            [
                [
                    "Step 1: Janet's ducks lay 16 eggs per day. She eats 3 eggs and bakes 4 eggs.",
                    "Step 2: So, she uses 3 + 4 = <<3+4=7>>7 eggs for eating and baking.",
                    "Step 3: She sells the remaining eggs, which is 16 - 7 = <<16-7=9>>9 duck eggs every day.",
                    "Step 4: She sells each egg for $2, so the total amount she makes is 9 * 2 = <<9*2=18>>18 dollars every day.",
                    "The answer is: 18",
                ],
                [
                    "Step 1: Janet's ducks lay 16 eggs per day.",
                    "Step 2: She eats 3 eggs and bakes 4 eggs, which is a total of 3 + 4 = <<3+4=7>>7 eggs.",
                    "Step 3: She sells the remaining eggs, which is 16 - 7 = <<16-7=9>>9 duck eggs every day.",
                    "Step 4: Since she sells each egg for $2, she makes 9 * 2 = <<9*2=18>>18 dollars every day.",
                    "The answer is: 18",
                ],
                [
                    "Step 1: Janet's ducks lay 16 eggs per day.",
                    "Step 2: She consumes 7 eggs for eating and baking, which means she has 16 - 7 = <<16-7=9>>9 eggs left.",
                    "Step 3: She sells each egg for $2, so she makes 9 * 2 = <<9*2=18>>18 dollars every day.",
                    "The answer is: 18",
                ],
            ],
        ]

        golden_answers = ["The answer is: 18", "The answer is: 18"]
        input_positions = [(0, 0, 0), (0, 1, 0), (0, 1, 1)]
        statistics = [
            {"input_tokens": [12], "output_tokens": [12]},
            {"input_tokens": [12], "output_tokens": [12]},
            {"input_tokens": [12], "output_tokens": [12]},
        ]
        raw_outputs = [[], []]
        raw_inputs = [
            {
                "role": "system",
                "content": 'You are a math teacher who helps students by breaking down word problems into clear, logical steps.\nWhen given a problem statement and any number of initial step, generate the remaining steps needed to reach the final answer.\nEach step should:\n\n- Build logically on previous steps\n- Explain the reasoning in natural language\n- Lead to the final answer\n- Multiple solution paths are acceptable\n- Steps should be concise but clear\n- Each calculation should be shown explicitly\n- The final answer must be clearly stated\n- The number of steps may vary based on the solution approach\n\n# Format requirements:\n- Each step should be numbered sequentially, continuing from the last given step\n- The final step should clearly state "The answer is: [result]"\n- Each step can use different approaches but must be mathematically valid\n\n# Rules:\n- All calculations must be shown within <<>> brackets\n- Basic operations: use * for multiplication, / for division, + for addition, - for subtraction\n- Write the full calculation and result, e.g., <<5*10=50>>50\n\n# Examples:\n## Input\nKrystian works in the library. He borrows an average of 40 books every day. Every Friday, his number of borrowed books is about 40% higher than the daily average. How many books does he borrow in a week if the library is open from Monday to Friday?\nStep 1: On Friday, Krystian borrows 40 * 0.4 = <<40*0.4=16>>16 more books than on a regular day.\n\n## Output 1\nStep 2: On Friday, Krystian borrows 40 + 16 = <<40+16=56>>56 books in total.\nStep 3: For the other 4 days (Monday to Thursday), he borrows 40 * 4 = <<40*4=160>>160 books.\nStep 4: The total books for the week is 160 + 56 = <<160+56=216>>216. The answer is: 216\n\n## Output 2\nStep 2: In total, he borrows 40 + 16 = <<40+16=56>>56 books on Friday.\nStep 3: For the whole week (4 regular days plus Friday), the total is (40 * 4) + 56 = <<(40*4)+56=216>>216. The answer is: 216\n\n## Output 3\nStep 2: On Friday, he borrows 40 + 40/100 * 40 = <<40+40/100*40=56>>56 books.\nStep 3: In a week, he borrows 5.7 * 7 = <<5.7*7=40>>40 books. The answer is: 40',
            },
            {
                "role": "user",
                "content": "Generate 3 example solutions to the same problem, separated by a single `---` and nothing else.\nResponse format:\n```\nStep i: step i explanation.\nStep i+1: step i+1 explanation.\nThe answer is: X\n\n---\n\nStep 2: step i explanation.\nStep 3: step i+1 explanation.\nThe answer is: Y\n```\n\nThis is the problem:\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
            },
        ]
        results = task._auto_label(
            inputs,
            final_outputs,
            input_positions,
            golden_answers,
            statistics,
            raw_outputs,
            raw_inputs,
        )
        assert results == [
            {
                "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "golden_solution": [
                    "Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.",
                    "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.",
                    "The answer is: 18",
                ],
                "solutions": [[], [], []],
                "model_name": "math-shepherd-completer",
                "distilabel_metadata": {
                    "statistics_math_shepherd_completer_0": {
                        "input_tokens": [12],
                        "output_tokens": [12],
                    }
                },
            }
        ]
