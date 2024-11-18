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
from typing import TYPE_CHECKING, Any, Dict, Final, Optional, Union

from jinja2 import Template
from pydantic import PositiveInt

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.math_shepherd.utils import split_solution_steps

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


SYSTEM_PROMPT = """\
You are a math tutor that helps students solve math problems by breaking them down into clear, logical steps. Follow these guidelines:

# For each step:
- Clearly explain the reasoning
- Show the calculated result for any arithmetic calculation
- Present intermediate calculations clearly
- Use clear, concise language to explain the mathematical reasoning

# Format requirements:
- Number each step starting with "Step 1:"
- The final step should clearly state "The answer is: [result]"
- Keep explanations clear and concise

{{ extra_rules }}{{ few_shots }}"""

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
Step 3: Subtract the discount from the regular price: 18 - 3.6 = <<18-3.6=14.4>>14.4 dollars. The answer is: 14.4

## Input
A recipe calls for 2.5 cups of flour to make 12 cookies. How many cups of flour are needed to make 30 cookies?

## Output
Step 1: Find out how many cups of flour are needed per cookie: 2.5 ÷ 12 = <<2.5/12=0.208333>>0.208333 cups
Step 2: Calculate the flour needed for 30 cookies: 0.208333 * 30 = <<0.208333*30=6.25>>6.25 cups. The answer is: 6.25
"""

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

TEMPLATE = """{% if M %}Generate {{ M }} example solutions to the following problem, separated by a single `---`:{% endif %}
{{ instruction }}"""


class MathShepherdGenerator(Task):
    """Math Shepherd solution generator.

    This task is in charge of generating completions for a given instruction, in the format expected
    by the Math Shepherd Completer task. The attributes make the task flexible to be used with different
    types of dataset and LLMs, but we provide examples for the GSM8K and MATH datasets as presented
    in the original paper. Before modifying them, review the current defaults to ensure the completions
    are generated correctly. This task can be used to generate the golden solutions for a given problem if
    not provided, as well as possible solutions to be then labeled by the Math Shepherd Completer.
    Only one of `solutions` or `golden_solution` will be generated, depending on the value of M.

    Attributes:
        system_prompt: The system prompt to be used in the completions. The default one has been
            checked and generates good completions using Llama 3.1 with 8B and 70B,
            but it can be modified to adapt it to the model and dataset selected.
            Take into account that the system prompt includes 2 variables in the Jinja2 template,
            {{extra_rules}} and {{few_shot}}. These variables are used to include extra rules, for example
            to steer the model towards a specific type of responses, and few shots to add examples.
            They can be modified to adapt the system prompt to the dataset and model used without needing
            to change the full system prompt.
        extra_rules: This field can be used to insert extra rules relevant to the type of dataset.
            For example, in the original paper they used GSM8K and MATH datasets, and this field
            can be used to insert the rules for the GSM8K dataset.
        few_shots: Few shots to help the model generating the completions, write them in the
            format of the type of solutions wanted for your dataset.
        M: Number of completions to generate for each step. By default is set to 1, which will
            generate the "golden_solution". In this case select a stronger model, as it will be used
            as the source of true during labelling. If M is set to a number greater than 1, the task
            will generate a list of completions to be labeled by the Math Shepherd Completer task.

    Input columns:
        - instruction (`str`): The task or instruction.

    Output columns:
        - golden_solution (`str`): The step by step solution to the instruction.
            It will be generated if M is equal to 1.
        - solutions (`str`): A JSON formatted list of possible solutions to the instruction.
            It will be generated if M is greater than 1.
        - model_name (`str`): The name of the model used to generate the revision.

    Categories:
        - text-generation

    References:
        - [`Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations`](https://arxiv.org/abs/2312.08935)

    Examples:
        Generate the solution for a given instruction (prefer a stronger model here):

        ```python
        from distilabel.steps.tasks import MathShepherdGenerator
        from distilabel.models import InferenceEndpointsLLM

        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            generation_kwargs={
                "temperature": 0.6,
                "max_new_tokens": 1024,
            },
        )
        task = MathShepherdGenerator(
            name="golden_solution_generator",
            llm=llm,
        )

        task.load()

        result = next(
            task.process(
                [
                    {
                        "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                    },
                ]
            )
        )
        # [[{'instruction': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        # 'golden_solution': '["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\\u2019s market.", "The answer is: 18"]'}]]
        ```

        Generate M completions for a given instruction (a less strong model is more helpful here):

        ```python
        from distilabel.steps.tasks import MathShepherdGenerator
        from distilabel.models import InferenceEndpointsLLM

        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            generation_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 2048,
            },
        )
        task = MathShepherdGenerator(
            name="solution_generator",
            llm=llm,
            M=2
        )

        task.load()

        result = next(
            task.process(
                [
                    {
                        "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                    },
                ]
            )
        )
        # [[{'instruction': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        # 'solutions': '[["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. -", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\\u2019s market.", "The answer is: 18"], ["Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking. +", "Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day. +", "Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18.", "The answer is: 18"]]'}]]
        ```
    """

    system_prompt: Optional[str] = SYSTEM_PROMPT
    extra_rules: Optional[str] = RULES_GSM8K
    few_shots: Optional[str] = FEW_SHOTS_GSM8K
    M: Optional[PositiveInt] = None

    def load(self) -> None:
        super().load()
        if self.system_prompt is not None:
            self.system_prompt = Template(self.system_prompt).render(
                extra_rules=self.extra_rules or "",
                few_shots=self.few_shots or "",
            )
        self._template = Template(TEMPLATE)

    @property
    def inputs(self) -> "StepColumns":
        return ["instruction"]

    @property
    def outputs(self) -> "StepColumns":
        return {
            "solutions": False,
            "golden_solution": False,
            "model_name": True,
        }

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        messages = [
            {
                "role": "user",
                "content": self._template.render(
                    instruction=input["instruction"],
                    M=self.M,
                ),
            }
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        output_name = "solutions" if self.M else "golden_solution"
        if output is None:
            input.update(**{output_name: None})
            return input

        if self.M:
            solutions = [split_solution_steps(o) for o in output.split("---")]
        else:
            solutions = split_solution_steps(output)

        input.update(**{output_name: json.dumps(solutions)})
        return input
