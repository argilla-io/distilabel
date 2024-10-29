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
import re
from typing import TYPE_CHECKING, Any, Dict, Final, Optional, Union

from jinja2 import Template
from pydantic import PositiveInt

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


SYSTEM_PROMPT = """\
You are a math tutor that helps students solve math problems by breaking them down into clear, logical steps{% if include_errors %}, but trick them by including unannounced errors in the reasoning without explaining them{% endif %}. Follow these guidelines:

# For each step:
- Clearly explain the reasoning
- Show the calculated result for any arithmetic calculation
- Present intermediate calculations clearly
- Use clear, concise language to explain the mathematical reasoning

# Format requirements:
- Number each step starting with "Step 1:"
- Include a clear "The answer is: " statement at the end of each problem
- Keep explanations clear and concise

{{ extra_rules }}{{ few_shots }}{{ errors }}"""

RULES_GSM8K: Final[str] = """\
# Rules:
- All calculations must be shown within <<>> brackets
- Basic operations: use * for multiplication, / for division, + for addition, - for subtraction
- Write the full calculation and result, e.g., <<5*10=50>>50
"""

FEW_SHOTS_GSM8K: Final[str] = """
# Examples:
## Input
A store sells notebooks for $3 each. If you buy 5 or more, you get a 20% discount. How much would you pay for 6 notebooks?

## Output
Step 1: Calculate the regular price for 6 notebooks: 6 * $3 = <<63=18>>18 dollars
Step 2: Calculate the 20% discount amount: 18 * 20/100 = <<1820/100=3.6>>3.6 dollars
Step 3: Subtract the discount from the regular price: 18 - 3.6 = <<18-3.6=14.4>>14.4 dollars. Answer: 14.4

## Input
A recipe calls for 2.5 cups of flour to make 12 cookies. How many cups of flour are needed to make 30 cookies?

## Output
Step 1: Find out how many cups of flour are needed per cookie: 2.5 ÷ 12 = <<2.5/12=0.208333>>0.208333 cups
Step 2: Calculate the flour needed for 30 cookies: 0.208333 * 30 = <<0.208333*30=6.25>>6.25 cups. Answer: 6.25"""

RULES_MATH: Final[str] = """\
# Rules:
- Always wrap mathematical expressions in $ symbols
- Use LaTeX-style math notation with $ symbols for mathematical expressions
- Format operations and equations properly using LaTeX notation within $ symbols
- Keep explanations precise and mathematically rigorous
- Use $\boxed{}$ notation only in the final step
"""

FEW_SHOTS_MATH: Final[str] = """
# Examples
## Input
Find the sum of the first three perfect squares greater than 50.

## Output
Step 1: The first perfect square greater than 50 is $8^2 = 64$.
Step 2: The second perfect square is $9^2 = 81$.
Step 3: The third perfect square is $10^2 = 100$.
Step 4: The sum is $64 + 81 + 100 = 245$.
Step 5: Therefore, the answer is $\boxed{245}$. The answer is: 245

## Input
What is the value of $2^5 + 3^3$?

## Output
Step 1: Calculate $2^5 = 32$.
Step 2: Calculate $3^3 = 27$.
Step 3: Add the results: $32 + 27 = 59$.
Step 4: Therefore, the answer is $\boxed{59}$. The answer is: 59
"""


TEMPLATE = """\
This is your instruction, don't include it in your answer:
{{ instruction }}{% if example_solutions %}{{ example_solutions }}{% endif %}"""


def split_solution_steps(text):
    """
    Split a step-by-step solution text into individual components.
    Returns a list of steps and the final answer.
    """
    # Pattern to match:
    # 1. Steps starting with "Step N:" and capturing all content until the next step or answer
    # 2. The final answer starting with "The answer is:"
    pattern = r"Step \d+:.*?(?=Step \d+:|The answer is:|$)|The answer is:.*"

    # Find all matches, strip whitespace
    matches = [match.strip() for match in re.findall(pattern, text, re.DOTALL)]

    return matches


class MathShepherdGenerator(Task):
    system_prompt: Optional[str] = SYSTEM_PROMPT
    extra_rules: Optional[str] = RULES_GSM8K
    few_shots: Optional[str] = FEW_SHOTS_GSM8K
    include_errors: bool = True
    N: PositiveInt = 1

    def load(self) -> None:
        super().load()
        errors = ""
        self._example_solutions = ""
        if self.include_errors:
            errors = "\n\nInclude errors to help students learn from their mistakes in any of the steps, including the final answer."
            self._example_solutions = (
                f"\n\nGenerate {self.N} example solution"
                if self.N == 1
                else f"\n\nGenerate {self.N} example solutions to the same problem, separated by a single `---`"
            )

        if self.system_prompt is not None:
            self.system_prompt = Template(self.system_prompt).render(
                extra_rules=self.extra_rules or "",
                few_shots=self.few_shots or "",
                errors=errors,
                include_errors=self.include_errors,
            )
        self._template = Template(TEMPLATE)

    @property
    def inputs(self) -> "StepColumns":
        return ["instruction"]

    @property
    def outputs(self) -> "StepColumns":
        return ["steps", "model_name"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        messages = [
            {
                "role": "user",
                "content": self._template.render(
                    instruction=input["instruction"],
                    example_solutions=self._example_solutions,
                ),
            }
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        if output is None:
            input.update(**{"steps": None})
            return input
        if self.include_errors:
            examples = [split_solution_steps(o) for o in output.split("---")]
        else:
            examples = [split_solution_steps(output)]
        input.update(**{"steps": json.dumps(examples)})
        return input
