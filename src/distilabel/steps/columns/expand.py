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

import json
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import field_validator, model_validator
from typing_extensions import Self

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
        encoded: A bool to inform Whether the columns are JSON encoded lists. If this value is
            set to True, the columns will be decoded before expanding. Alternatively, to specify
            columns that can be encoded, a list can be provided. In this case, the column names
            informed must be a subset of the columns selected for expansion.
        split_statistics: A bool to inform whether the statistics in the `distilabel_metadata`
            column should be split into multiple rows.
            If we want to expand some columns containing a list of strings that come from
            having parsed the output of an LLM, the tokens in the `statistics_{step_name}`
            of the `distilabel_metadata` column should be splitted to avoid multiplying
            them if we aggregate the data afterwards. For example, with a task that is supposed
            to generate a list of N instructions, and we want each of those N instructions in
            different rows, we should split the statistics by N.
            In such a case, set this value to True.

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

        Expand the selected columns which are JSON encoded into multiple rows:

        ```python
        from distilabel.steps import ExpandColumns

        expand_columns = ExpandColumns(
            columns=["generation"],
            encoded=True,  # It can also be a list of columns that are encoded, i.e. ["generation"]
        )
        expand_columns.load()

        result = next(
            expand_columns.process(
                [
                    {
                        "instruction": "instruction 1",
                        "generation": '["generation 1", "generation 2"]'}
                ],
            )
        )
        # >>> result
        # [{'instruction': 'instruction 1', 'generation': 'generation 1'}, {'instruction': 'instruction 1', 'generation': 'generation 2'}]
        ```

        Expand the selected columns and split the statistics in the `distilabel_metadata` column:

        ```python
        from distilabel.steps import ExpandColumns

        expand_columns = ExpandColumns(
            columns=["generation"],
            split_statistics=True,
        )
        expand_columns.load()

        result = next(
            expand_columns.process(
                [
                    {
                        "instruction": "instruction 1",
                        "generation": ["generation 1", "generation 2"],
                        "distilabel_metadata": {
                            "statistics_generation": {
                                "input_tokens": [12],
                                "output_tokens": [12],
                            },
                        },
                    }
                ],
            )
        )
        # >>> result
        # [{'instruction': 'instruction 1', 'generation': 'generation 1', 'distilabel_metadata': {'statistics_generation': {'input_tokens': [6], 'output_tokens': [6]}}}, {'instruction': 'instruction 1', 'generation': 'generation 2', 'distilabel_metadata': {'statistics_generation': {'input_tokens': [6], 'output_tokens': [6]}}}]
        ```
    """

    columns: Union[Dict[str, str], List[str]]
    encoded: Union[bool, List[str]] = False
    split_statistics: bool = False

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

    @model_validator(mode="after")
    def is_subset(self) -> Self:
        """Ensure the "encoded" column names are a subset of the "columns" selected.

        Returns:
            The "encoded" attribute updated to work internally.
        """
        if isinstance(self.encoded, list):
            if not set(self.encoded).issubset(set(self.columns.keys())):
                raise ValueError(
                    "The 'encoded' columns must be a subset of the 'columns' selected for expansion."
                )
        if isinstance(self.encoded, bool):
            self.encoded = list(self.columns.keys()) if self.encoded else []
        return self

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
        if self.encoded:
            for input in inputs:
                for column in self.encoded:
                    input[column] = json.loads(input[column])

        yield [row for input in inputs for row in self._expand_columns(input)]

    def _expand_columns(self, input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand the columns in the input data.

        Args:
            input: The input data.

        Returns:
            The expanded rows.
        """
        metadata_visited = False
        expanded_rows = []
        # Update the columns here to avoid doing the validation on the `inputs`, as the
        # `distilabel_metadata` is not defined on Pipeline creation on the DAG.
        columns = self.columns
        if self.split_statistics:
            columns["distilabel_metadata"] = "distilabel_metadata"

        for expand_column, new_column in columns.items():  # type: ignore
            data = input.get(expand_column)
            input, metadata_visited = self._split_metadata(
                input, len(data), metadata_visited
            )

            rows = []
            for item, expanded in zip_longest(*[data, expanded_rows], fillvalue=input):
                rows.append({**expanded, new_column: item})
            expanded_rows = rows
        return expanded_rows

    def _split_metadata(
        self, input: Dict[str, Any], n: int, metadata_visited: bool = False
    ) -> None:
        """Help method to split the statistics in `distilabel_metadata` column.

        Args:
            input: The input data.
            n: Number of splits to apply to the tokens (if we have 12 tokens and want to split
                them 3 times, n==3).
            metadata_visited: Bool to prevent from updating the data more than once.

        Returns:
            Updated input with the `distilabel_metadata` updated.
        """
        # - If we want to split the statistics, we need to ensure that the metadata is present.
        # - Metadata can only be visited once per row to avoid successive splitting.
        # TODO: For an odd number of tokens, this will miss 1, we have to fix it.
        if (
            self.split_statistics
            and (metadata := input.get("distilabel_metadata", {}))
            and not metadata_visited
        ):
            for k, v in metadata.items():
                if k.startswith("statistics_") and (
                    "input_tokens" in v and "output_tokens" in v
                ):
                    # For num_generations>1 we assume all the tokens should be divided by n
                    input_tokens = [value // n for value in v["input_tokens"]]
                    output_tokens = [value // n for value in v["output_tokens"]]
                    input["distilabel_metadata"][k] = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                metadata_visited = True
            # Once we have updated the metadata, Create a list out of it to let the
            # following section to expand it as any other column.
            if isinstance(input["distilabel_metadata"], dict):
                input["distilabel_metadata"] = [input["distilabel_metadata"]] * n
        return input, metadata_visited
