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

from distilabel.pipeline.utils import combine_dicts
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class CombineColumns(Step):
    """CombineColumns is a Step that implements the `process` method that calls the `combine_dicts`
    function to handle and combine a list of `StepInput`. Also `CombineColumns` provides two attributes
    `merge_columns` and `output_merge_columns` to specify the columns to merge and the output columns
    which will override the default value for the properties `inputs` and `outputs`, respectively.
    """

    merge_columns: List[str]
    output_merge_columns: Optional[List[str]] = None

    @property
    def inputs(self) -> List[str]:
        return self.merge_columns

    @property
    def outputs(self) -> List[str]:
        return (
            self.output_merge_columns
            if self.output_merge_columns is not None
            else [f"merged_{column}" for column in self.merge_columns]
        )

    def process(self, *args: StepInput) -> "StepOutput":
        yield combine_dicts(
            *args,
            merge_keys=set(self.inputs),
            output_merge_keys=set(self.outputs),
        )
