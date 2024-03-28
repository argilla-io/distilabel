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

from typing import TYPE_CHECKING, List

from typing_extensions import override

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class KeepColumns(Step):
    """KeepColumns is a Step that implements the `process` method that keeps only the columns
    specified in the `columns` attribute. Also `KeepColumns` provides an attribute `columns` to
    specify the columns to keep which will override the default value for the properties `inputs`
    and `outputs`.

    Note:
        The order in which the columns are provided is important, as the output will be sorted
        using the provided order, which is useful before pushing either a `dataset.Dataset` via
        the `PushToHub` step or a `distilabel.Distiset` via the `Pipeline.run` output variable.

    Attributes:
        columns: List of strings with the names of the columns to keep.

    Input columns:
        - dynamic, based on the `columns` value provided.

    Output columns:
        - dynamic, based on the `columns` value provided.
    """

    columns: List[str]

    @property
    def inputs(self) -> List[str]:
        """The inputs for the task are the column names in `columns`."""
        return self.columns

    @property
    def outputs(self) -> List[str]:
        """The outputs for the task are the column names in `columns`."""
        return self.columns

    @override
    def process(self, *inputs: StepInput) -> "StepOutput":
        """The `process` method keeps only the columns specified in the `columns` attribute.

        Args:
            *inputs: A list of dictionaries with the input data.

        Yields:
            A list of dictionaries with the output data.
        """
        for input in inputs:
            outputs = []
            for item in input:
                outputs.append({col: item[col] for col in self.columns})
            yield outputs
