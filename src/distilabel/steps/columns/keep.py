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

from typing import TYPE_CHECKING, Any, List

from typing_extensions import override

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class KeepColumns(Step):
    """Keeps selected columns in the dataset.

    `KeepColumns` is a `Step` that implements the `process` method that keeps only the columns
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
        - dynamic (determined by `columns` attribute): The columns to keep.

    Output columns:
        - dynamic (determined by `columns` attribute): The columns that were kept.

    Categories:
        - columns

    Examples:
        Select the columns to keep:

        ```python
        from distilabel.steps import KeepColumns

        keep_columns = KeepColumns(
            columns=["instruction", "generation"],
        )
        keep_columns.load()

        result = next(
            keep_columns.process(
                [{"instruction": "What's the brightest color?", "generation": "white", "model_name": "my_model"}],
            )
        )
        # >>> result
        # [{'instruction': "What's the brightest color?", 'generation': 'white'}]
        ```
    """

    columns: List[str]

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.inputs = self.columns
        self.outputs = self.columns

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
