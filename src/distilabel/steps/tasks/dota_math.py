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
        queries = [match.group(2).strip() for match in matches]
        return {"augmented_queries": queries}
