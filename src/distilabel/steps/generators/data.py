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

from typing import TYPE_CHECKING, Any, Dict, List

from typing_extensions import override

from distilabel.steps.base import (
    GeneratorStep,
)

if TYPE_CHECKING:
    from distilabel.steps.typing import GeneratorStepOutput


class LoadData(GeneratorStep):
    """A generator step that loads a dataset from a list of dictionaries.

    This step will load the dataset and yield the transformed data as it is loaded from the list of dictionaries.

    Runtime parameters:

    - `batch_size`: The batch size to use when processing the data.

    Columns:
    - `input`: None
    - `output`: dynamic, based on the keys found on the first dictionary of the list
    """

    data: List[Dict[str, Any]]

    @override
    def process(self, offset: int = 0) -> "GeneratorStepOutput":  # type: ignore
        """Yields batches from the a dataset from a list of dictionaries.

        Args:
            offset: The offset to start the generation from. Defaults to 0.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """
        if offset:
            self.data = self.data[offset:]
        
        while self.data:
            batch = self.data[: self.batch_size]
            self.data = self.data[self.batch_size :]
            yield (
                batch,
                True if len(self.data) == 0 else False,
            )

    @property
    def outputs(self) -> List[str]:
        """List of strings with the names of the columns that the step will produce as
        output.

        Returns:
            List of strings with the names of the columns that the step will produce as
            output.
        """
        return list(self.data[0].keys())
