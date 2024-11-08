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

from typing_extensions import override

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


class AutoEvolFailureDetector(Step):
    """Detects failures in generated instructions, following heuristics from appendix A in the paper.

    Input columns:
        - answers (`str`): List with arguments to be passed to the function,
            dumped as a string from a list of dictionaries. Should be loaded using
            `json.loads`.

    Output columns:
        - keep_row_after_execution_check (`bool`): Whether the function should be kept or not.
        - execution_result (`str`): The result from executing the function.

    Categories:
        - filtering

    References:
        - [Automatic Instruction Evolving for Large Language Models](https://arxiv.org/abs/2406.00770)
        - [arcee-ai/EvolKit](https://github.com/arcee-ai/EvolKit)

    Examples:
        Detects failures in generated instructions:

        ```python
        from distilabel.steps.tasks import AutoEvolFailureDetector

        task = AutoEvolFailureDetector()
        task.load()

        res = next(
            task.process(
                [
                    {
                        "instruction": "blah blah blah"
                    }
                ]
            )
        )
        # ...
        ```
    """

    def load(self) -> None:
        super().load()
        self.stagnant_pattern = re.compile(
            r"\b(understood|thank you|noted|got it|okay|alright)\b.*\?$", re.IGNORECASE
        )
        self.insufficient_pattern = re.compile(
            r"\b(sure|certainly|of course|happy to help)\b.*\?$|what do you mean|could you explain",
            re.IGNORECASE,
        )
        self.loss_pattern = re.compile(
            r"please provide|need more information|could you clarify|what exactly",
            re.IGNORECASE,
        )

    @property
    def inputs(self) -> "StepColumns":
        return ["instruction"]

    @property
    def outputs(self) -> "StepColumns":
        return ["keep_row_after_failure_detection"]

    def is_failure(self, response: str) -> bool:
        return (
            bool(self.stagnant_pattern.search(response))
            or bool(self.insufficient_pattern.search(response))
            or bool(self.loss_pattern.search(response))
        )

    @override
    def process(self, *inputs: StepInput) -> "StepOutput":
        """The `process` method keeps only the columns specified in the `columns` attribute.

        Args:
            *inputs: A list of dictionaries with the input data.

        Yields:
            A list of dictionaries with the output data.
        """
        for input in inputs:
            input["keep_row_after_failure_detection"] = self.is_failure(
                input["instruction"]
            )

        yield inputs
