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
from typing import TYPE_CHECKING

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


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


class FormatPRM(Step):
    """Helper step to transform the data into the format expected by the PRM model.

    Following the format presented in [peiyi9979/Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd?row=0),
    this step creates the columns input and label, where the input is the instruction
    with the solution (and the tag replaced by a token), and the label is the instruction
    with the solution, both separated by a newline.

    Attributes:
        step_token (str): String that serves as a unique token denoting the position
            for predicting the step score.

    Input columns:
        - instruction (`str`): The task or instruction.
        - solutions (`list[str]`): List of steps with a solution to the task.

    Output columns:
        - input (`str`): The instruction with the solutions, where the label tags
            are replaced by a token.
        - label (`str`): The instruction with the solutions.

    Categories:
        - text-manipulation
        - columns

    References:
        - [`Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations`](https://arxiv.org/abs/2312.08935)
        - [peiyi9979/Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd?row=0)

    Examples:
        Prepare your data to train a PRM model:

        ```python
        from distilabel.steps.tasks import FormatPRM
        from distilabel.steps import ExpandColumns

        expand_columns = ExpandColumns(
            columns=["solutions"],
            encoded=True,
        )
        expand_columns.load()

        # Define our PRM formatter
        formatter = FormatPRM()
        formatter.load()

        # Expand the solutions column as it comes from the MathShepherdCompleter
        result = next(
            expand_columns.process(
                [
                    {
                        "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                        "solutions": '[["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"]]'
                    },
                ]
            )
        )
        result = next(formatter.process(result))
        # result[0]["input"]
        # "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. ки\nStep 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. ки\nStep 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 ки"
        # result[0]["label"]
        # "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +\nStep 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +\nStep 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"
        ```

    Citations:

        ```
        @misc{wang2024mathshepherdverifyreinforcellms,
            title={Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations},
            author={Peiyi Wang and Lei Li and Zhihong Shao and R. X. Xu and Damai Dai and Yifei Li and Deli Chen and Y. Wu and Zhifang Sui},
            year={2024},
            eprint={2312.08935},
            archivePrefix={arXiv},
            primaryClass={cs.AI},
            url={https://arxiv.org/abs/2312.08935},
        }
        ```
    """

    step_token: str = "ки"

    @property
    def inputs(self) -> "StepColumns":
        return ["instruction", "solutions"]

    @property
    def outputs(self) -> "StepColumns":
        return ["input", "label"]

    def process(self, inputs: StepInput) -> "StepOutput":
        """The process prepares the data for the `APIGenGenerator` task.

        If a single example is provided, it is copied to avoid raising an error.

        Args:
            inputs: A list of dictionaries with the input data.

        Yields:
            A list of dictionaries with the output data.
        """
        for input in inputs:
            instruction = input["instruction"]
            solution = input["solutions"]
            replaced = [step[:-1] + self.step_token for step in solution]

            input["input"] = instruction + " " + "\n".join(replaced)
            input["label"] = instruction + " " + "\n".join(solution)

        yield inputs
