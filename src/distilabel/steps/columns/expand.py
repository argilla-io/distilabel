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

from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import field_validator

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


class ExpandColumns(Step):
    """Expand columns that contain lists into multiple rows.

    `ExpandColumns` is a `Step` that takes a list of columns and expands them into multiple
    rows. The new rows will have the same data as the original row, except for the expanded
    column, which will contain a single item from the original list.

    Attributes:
        columns: A dictionary that maps the column to be expanded to the new column name
            or a list of columns to be expanded. If a list is provided, the new column name
            will be the same as the column name.

    Input columns:
        - dynamic (determined by `columns` attribute): The columns to be expanded into
            multiple rows.

    Output columns:
        - dynamic (determined by `columns` attribute):  The expanded columns.

    Categories:
        - columns

    Examples:
        Expand the selected columns into multiple rows:

        ```python
        from distilabel.steps import ExpandColumns

        expand_columns = ExpandColumns(
            columns=["generation"],
        )
        expand_columns.load()

        result = next(
            expand_columns.process(
                [
                    {
                        "instruction": "instruction 1",
                        "generation": ["generation 1", "generation 2"]}
                ],
            )
        )
        # >>> result
        # [{'instruction': 'instruction 1', 'generation': 'generation 1'}, {'instruction': 'instruction 1', 'generation': 'generation 2'}]
        ```
    """

    columns: Union[Dict[str, str], List[str]]

    @field_validator("columns")
    @classmethod
    def always_dict(cls, value: Union[Dict[str, str], List[str]]) -> Dict[str, str]:
        """Ensure that the columns are always a dictionary.

        Args:
            value: The columns to be expanded.

        Returns:
            The columns to be expanded as a dictionary.
        """
        if isinstance(value, list):
            return {col: col for col in value}

        return value

    @property
    def inputs(self) -> "StepColumns":
        """The columns to be expanded."""
        return list(self.columns.keys())

    @property
    def outputs(self) -> "StepColumns":
        """The expanded columns."""
        return [
            new_column if new_column else expand_column
            for expand_column, new_column in self.columns.items()
        ]

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Expand the columns in the input data.

        Args:
            inputs: The input data.

        Yields:
            The expanded rows.
        """
        yield [row for input in inputs for row in self._expand_columns(input)]

    def _expand_columns(self, input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand the columns in the input data.

        Args:
            input: The input data.

        Returns:
            The expanded rows.
        """
        expanded_rows = []
        for expand_column, new_column in self.columns.items():  # type: ignore
            data = input.get(expand_column)
            rows = []
            for item, expanded in zip_longest(*[data, expanded_rows], fillvalue=input):
                rows.append({**expanded, new_column: item})
            expanded_rows = rows
        return expanded_rows
