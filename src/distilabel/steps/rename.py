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

from typing import Any, Dict, Generator, List

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import Step, StepInput


class RenameColumns(Step):
    """RenameColumns is a Step that implements the `process` method that renames the columns
    specified in the `rename_mappings` attribute. This can be handy if a dataset has a column
    named differently than expected by the downstream tasks.

    Args:
        rename_mappings: Dict from str (original column) to str (new column).

    Input columns:
        - dynamic, based on the `rename_mappings` value provided.

    Output columns:
        - dynamic, based on the `rename_mappings` value provided.
    """

    rename_mappings: RuntimeParameter[Dict[str, str]]

    @property
    def inputs(self) -> List[str]:
        """There are no inputs."""
        return []

    @property
    def outputs(self) -> List[str]:
        """The outputs for the task are the column names in `rename_mappings` values."""
        return list(self.rename_mappings.values())  # type: ignore

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
        """The `process` method keeps only the columns specified in the `columns` attribute.

        Args:
            inputs: A dictionary with the columns to update.

        Yields:
            A list of dictionaries with the output data.
        """
        outputs = []
        for input in inputs:
            outputs.append(
                {self.rename_mappings.get(k, k): v for k, v in input.items()}  # type: ignore
            )
        yield outputs
