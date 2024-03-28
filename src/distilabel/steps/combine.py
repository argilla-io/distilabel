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

from typing import TYPE_CHECKING, List, Optional

from typing_extensions import override

from distilabel.pipeline.utils import combine_dicts
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class CombineColumns(Step):
    """CombineColumns is a Step that implements the `process` method that calls the `combine_dicts`
    function to handle and combine a list of `StepInput`. Also `CombineColumns` provides two attributes
    `columns` and `output_columns` to specify the columns to merge and the output columns
    which will override the default value for the properties `inputs` and `outputs`, respectively.

    Attributes:
        columns: List of strings with the names of the columns to merge.
        output_columns: Optional list of strings with the names of the output columns.

    Input columns:
        - dynamic, based on the `columns` value provided.

    Output columns:
        - dynamic, based on the `output_columns` value provided or `merged_{column}` for each column in `columns`.
    """

    columns: List[str]
    output_columns: Optional[List[str]] = None

    @property
    def inputs(self) -> List[str]:
        """The inputs for the task are the column names in `columns`."""
        return self.columns

    @property
    def outputs(self) -> List[str]:
        """The outputs for the task are the column names in `output_columns` or
        `merged_{column}` for each column in `columns`."""
        return (
            self.output_columns
            if self.output_columns is not None
            else [f"merged_{column}" for column in self.columns]
        )

    @override
    def process(self, *inputs: StepInput) -> "StepOutput":
        """The `process` method calls the `combine_dicts` function to handle and combine a list of `StepInput`.

        Args:
            *inputs: A list of `StepInput` to be combined.

        Yields:
            A `StepOutput` with the combined `StepInput` using the `combine_dicts` function.
        """
        yield combine_dicts(
            *inputs,
            merge_keys=self.inputs,
            output_merge_keys=self.outputs,
        )
