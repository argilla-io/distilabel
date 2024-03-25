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
    from distilabel.steps.typing import StepOutput


class ExpandColumns(Step):
    columns: Union[Dict[str, str], List[str]]

    @field_validator("columns")
    @classmethod
    def always_dict(cls, value: Union[Dict[str, str], List[str]]) -> Dict[str, str]:
        if isinstance(value, list):
            return {col: col for col in value}

        return value

    @property
    def outputs(self) -> List[str]:
        return [
            new_column if new_column else expand_column
            for expand_column, new_column in self.columns.items()
        ]

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        yield [row for input in inputs for row in self._expand_columns(input)]

    def _expand_columns(self, input: Dict[str, Any]) -> List[Dict[str, Any]]:
        expanded_rows = []
        for expand_column, new_column in self.columns.items():  # type: ignore
            data = input.get(expand_column)
            rows = []
            for item, expanded in zip_longest(*[data, expanded_rows], fillvalue=input):
                rows.append({**expanded, new_column: item})
            expanded_rows = rows
        return expanded_rows
