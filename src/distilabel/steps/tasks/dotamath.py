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

import re
from typing import TYPE_CHECKING, Any, Dict, Union

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import FormattedInput
    from distilabel.steps.typing import StepColumns

# Aka "Augmentation Prompt"
DOTA_MATH_AUGMENTATION_SYSTEM_PROMPT = """
I want you to act as a math teacher. You should think of some ways to help students do variation training for challenging competition mathematics problems.
Here are some ways you can refer: Introduce fractions or percentages, Combine multiple concepts, Include a conditional statement, Increase the complexity of the problem and so on. Response with specific format, like:
Introduce fractions or percentages: ##1 new question1 ##1
Combine multiple concepts: ##2 new question2 ##2
...
Increase the complexity of the problem: ##10: new question10 ##10
The nth problem must be strictly limited to between ##n and ##n for our subsequent regular extraction.
Now you are give a math problem, think for 10 different ways.
Given new problem:
{query}
""".lstrip()

DOTA_MATH_EXTRACT_AUGMENTED_QUERY_REGEX = re.compile(
    r"##(\d+)\s*(.*?)\s*##\1", re.DOTALL
)


class DotaMathAugmentQuery(Task):
    """Create new math queries from a seed one.

    `DotaMathAugmentQuery` is a `Task` that given a math query uses an `LLM` to create
    new augmented queries that can be more complex, contain more conditions, etc.
    """

    @property
    def inputs(self) -> "StepColumns":
        return ["query"]

    def format_input(self, input: Dict[str, Any]) -> "FormattedInput":
        query = input["query"]
        return [
            {
                "role": "user",
                "content": DOTA_MATH_AUGMENTATION_SYSTEM_PROMPT.format(query=query),
            }
        ]

    @property
    def outputs(self) -> "StepColumns":
        return ["augmented_queries"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        if output is None:
            return {"augmented_queries": None}

        matches = DOTA_MATH_EXTRACT_AUGMENTED_QUERY_REGEX.finditer(output)
        queries = []
        for match in matches:
            augmented_query = match.group(2).strip()
            # Sometimes the LLM includes the description of the augmentation applied before
            # giving the augmented query: "Increase the complexity of the problem: ..."
            augmented_query = augmented_query.split(":")[-1].strip()
            queries.append(augmented_query)
        return {"augmented_queries": queries}


# Aka "Generative Prompt"
DOTA_MATH_GENERATE_SOLUTION_SYSTEM_PROMPT = r"""
You are an exceptionally strong competitor in both math and programming contests, proficient in a wide range of mathematical knowledge and skilled in Python programming. Your command of Pre-algebra, Algebra, Number Theory, Counting and Probability, Geometry, Intermediate Algebra, and Precalculus is unparalleled. Your thinking is meticulous and profound, and the code you write always runs flawlessly and without error.

Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

1. Break the problem into subtasks.
2. Write functions to solve the problem; the function should not take any arguments.
3. Print the results of every subtask in the Python code, using the intermediate variables in Python programs to represent intermediate results, refer to the example below.
4. When writing the python program, avoid using decimal. Utilize functions from sympy and other necessary Python library, and simplify all fractions and square roots without converting them to decimal values.
5. Print the final answer on the last line.

Here is an example you may refer to:

*Problem:* Let

$$
f(x) = \begin{cases}
    ax + 3,    & \text{if } x > 2, \\
    x - 5,     & \text{if } -2 \leq x \leq 2, \\
    2x - b,    & \text{if } x < -2.
\end{cases}
$$

Find $a + b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).

*Solution:* We can decompose this problem into following sub-tasks:

1. Solve for $a$ by equating $ax + 3$ to $x - 5$ at $x = 2$.
2. Solve for $b$ by equating $x - 5$ to $2x - b$ at $x = -2$.
3. Add the values of $a$ and $b$ together to find the sum.

```python
from sympy import symbols, Eq, solve

def sum_a_and_b():
    a = symbols('a')
    b = symbols('b')
    equation1 = Eq(a * 2 + 3, 2 - 5)
    equation2 = Eq(-2 - 5, 2*(-2) - b)
    solution_a = solve(equation1, a)
    solution_b = solve(equation2, b)
    sum_ab = solution_a[0] + solution_b[0]
    # print the results of every subtask
    print(f"Equating the function at x = 2 gives us the equation {equation1}.")
    print(f"Solving this equation gives us the value of a: a = {solution_a[0]}.")
    print(f"Equating the function at x = -2 gives us the equation {equation2}.")
    print(f"Solving this equation gives us the value of b: b = {solution_b[0]}.")
    print(f"hence, a + b equals to {solution_a[0]}+{solution_b[0]} = {sum_ab}.")
    return sum_ab

sum_ab = sum_a_and_b()
# print the final answer
print(sum_ab)
```

```output
Output:
Equating the function at $x = 2$ gives us the equation $2a + 3 = -3$.
Solving this equation gives us the value of $a$: $a = -3$.
Equating the function at $x = -2$ gives us the equation $-7 = -b - 4$.
Solving this equation gives us the value of $b$: $b = 3$.
hence, $a + b$ equals to $-3+3 = 0$.
0
```

We find that the sum of $a$ and $b$ is $0$. This ensures the piecewise function is continuous across its entire domain. Therefore, the final answer is $\boxed{0}$.
""".lstrip()


class DotaMathSolutionGenerator(Task):
    @property
    def inputs(self) -> "StepColumns":
        return ["query"]

    def format_input(self, input: Dict[str, Any]) -> "FormattedInput":
        query = input["query"]
        return [
            {
                "role": "user",
                "content": DOTA_MATH_GENERATE_SOLUTION_SYSTEM_PROMPT.format(
                    query=query
                ),
            }
        ]

    @property
    def outputs(self) -> "StepColumns":
        return ["solution"]
