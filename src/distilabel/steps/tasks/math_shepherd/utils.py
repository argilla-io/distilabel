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
from typing import TYPE_CHECKING, Any, Literal, Union

import orjson

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.typing import StepColumns, StepOutput


def split_solution_steps(text: str) -> list[str]:
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

    This step can be used to format the data in one of 2 formats:
    Following the format presented
    in [peiyi9979/Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd?row=0),
    in which case this step creates the columns input and label, where the input is the instruction
    with the solution (and the tag replaced by a token), and the label is the instruction
    with the solution, both separated by a newline.
    Following TRL's format for training, which generates the columns prompt, completions, and labels.
    The labels correspond to the original tags replaced by boolean values, where True represents
    correct steps.

    Attributes:
        format: The format to use for the PRM model.
            "math-shepherd" corresponds to the original paper, while "trl" is a format
            prepared to train the model using TRL.
        step_token: String that serves as a unique token denoting the position
            for predicting the step score.
        tags: List of tags that represent the correct and incorrect steps.
            This only needs to be informed if it's different than the default in
            `MathShepherdCompleter`.

    Input columns:
        - instruction (`str`): The task or instruction.
        - solutions (`list[str]`): List of steps with a solution to the task.

    Output columns:
        - input (`str`): The instruction with the solutions, where the label tags
            are replaced by a token.
        - label (`str`): The instruction with the solutions.
        - prompt (`str`): The instruction with the solutions, where the label tags
            are replaced by a token.
        - completions (`List[str]`): The solution represented as a list of steps.
        - labels (`List[bool]`): The labels, as a list of booleans, where True represents
            a good response.

    Categories:
        - text-manipulation
        - columns

    References:
        - [`Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations`](https://arxiv.org/abs/2312.08935)
        - [peiyi9979/Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd?row=0)

    Examples:
        Prepare your data to train a PRM model with the Math-Shepherd format:

        ```python
        from distilabel.steps.tasks import FormatPRM
        from distilabel.steps import ExpandColumns

        expand_columns = ExpandColumns(columns=["solutions"])
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
                        "solutions": [["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"]]
                    },
                ]
            )
        )
        result = next(formatter.process(result))
        ```

        Prepare your data to train a PRM model with the TRL format:

        ```python
        from distilabel.steps.tasks import FormatPRM
        from distilabel.steps import ExpandColumns

        expand_columns = ExpandColumns(columns=["solutions"])
        expand_columns.load()

        # Define our PRM formatter
        formatter = FormatPRM(format="trl")
        formatter.load()

        # Expand the solutions column as it comes from the MathShepherdCompleter
        result = next(
            expand_columns.process(
                [
                    {
                        "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                        "solutions": [["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"]]
                    },
                ]
            )
        )

        result = next(formatter.process(result))
        # {
        #     "instruction": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        #     "solutions": [
        #         "Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +",
        #         "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +",
        #         "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"
        #     ],
        #     "prompt": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        #     "completions": [
        #         "Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required.",
        #         "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber.",
        #         "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3"
        #     ],
        #     "labels": [
        #         true,
        #         true,
        #         true
        #     ]
        # }
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

    format: Literal["math-shepherd", "trl"] = "math-shepherd"
    step_token: str = "ки"
    tags: list[str] = ["+", "-"]

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.format == "math-shepherd":
            self._formatter = self._format_math_shepherd
        else:
            self._formatter = self._format_trl

    @property
    def inputs(self) -> "StepColumns":
        return ["instruction", "solutions"]

    @property
    def outputs(self) -> "StepColumns":
        if self.format == "math-shepherd":
            return ["input", "label"]
        return ["prompt", "completions", "labels"]

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """The process prepares the data for the `APIGenGenerator` task.

        If a single example is provided, it is copied to avoid raising an error.

        Args:
            inputs: A list of dictionaries with the input data.

        Yields:
            A list of dictionaries with the output data.
        """
        for input in inputs:
            self._formatter(input)

        yield inputs  # type: ignore

    def _format_math_shepherd(
        self, input: dict[str, str]
    ) -> dict[str, Union[str, list[str]]]:
        instruction = input["instruction"]
        replaced = []
        # At this stage, the "solutions" column can only contain a single solution,
        # and the last item of each solution is the tag.
        solution = input["solutions"]
        for step in solution:
            # Check there's a string, because the step that generated
            # the solutions could have failed, and we would have an empty list.
            replaced.append(step[:-1] + self.step_token if len(step) > 1 else step)

        input["input"] = instruction + " " + "\n".join(replaced)
        input["label"] = instruction + " " + "\n".join(solution)

        return input  # type: ignore

    def _format_trl(
        self, input: dict[str, str]
    ) -> dict[str, Union[str, list[str], list[bool]]]:
        input["prompt"] = input["instruction"]
        completions: list[str] = []
        labels: list[bool] = []
        for step in input["solutions"]:
            token = step[-1]
            completions.append(step[:-1].strip())
            labels.append(True if token == self.tags[0] else False)

        input["completions"] = completions  # type: ignore
        input["labels"] = labels  # type: ignore

        return input  # type: ignore


def parse_json_response(json_str: str) -> Union[dict[str, Any], None]:
    """Helper function to clean and parse JSON strings generated by LLMs.
    Some common errors may appear (see the REPLACEMENTS dictionary) that need to be fixed before parsing,
    but the JSON is valid otherwise.
    """

    try:
        # First try parsing as-is
        return orjson.loads(json_str)
    except orjson.JSONDecodeError:
        # Apply all replacements
        for old, new in REPLACEMENTS.items():
            json_str = json_str.replace(old, new)

        try:
            # Try parsing after replacements
            return orjson.loads(json_str)
        except orjson.JSONDecodeError:
            # If still failing, try more aggressive cleaning

            # Remove any non-ASCII characters
            json_str = re.sub(r"[^\x00-\x7F]+", "", json_str)

            # Remove any remaining escape sequences except valid ones
            json_str = re.sub(r'\\([^"\\\/bfnrt])', r"\1", json_str)

            try:
                return orjson.loads(json_str)
            except orjson.JSONDecodeError:
                # Failed to parse JSON after all cleaning attempts
                return None


# Dictionary of common LLM JSON artifacts and their replacements
REPLACEMENTS: dict[str, str] = {
    # Escape sequence issues
    "\\)": ")",  # Incorrectly escaped parentheses
    "\\]": "]",  # Incorrectly escaped brackets
    "\\}": "}",  # Incorrectly escaped braces
    "\\`": "`",  # Incorrectly escaped backticks
    "\\'": "'",  # Incorrectly escaped single quotes
    '\\\\"': '\\"',
    '\\"': '"',  # Incorrectly escaped double quotes
    "\\\\n": "\\n",  # Double escaped newlines
    "\\\\t": "\\t",  # Double escaped tabs
    "\\\\r": "\\r",  # Double escaped carriage returns
    # # Markdown artifacts
    # '```json\n': '',     # Markdown code block start
    # '\n```': '',         # Markdown code block end
    # '`': '',             # Inline code markers
    # Common mathematical symbols that might be escaped
    "\\<": "<",  # Less than
    "\\>": ">",  # Greater than
    "\\=": "=",  # Equals
    "\\+": "+",  # Plus
    "\\-": "-",  # Minus
    "\\*": "*",  # Asterisk
    "\\|": "|",  # Pipe
    # Unicode escaping issues
    "\\u0022": '"',  # Double quote
    "\\u0027": "'",  # Single quote
    "\\u005C": "\\",  # Backslash
    # # Other common issues
    # '\n\n': '\n',       # Multiple newlines
    # '\t\t': '\t',       # Multiple tabs
}
