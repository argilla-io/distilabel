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

from pydantic import Field
from typing_extensions import override

from distilabel.steps.base import GeneratorStep

if TYPE_CHECKING:
    from distilabel.typing import GeneratorStepOutput


class LoadDataFromDicts(GeneratorStep):
    """Loads a dataset from a list of dictionaries.

    `GeneratorStep` that loads a dataset from a list of dictionaries and yields it in
    batches.

    Attributes:
        data: The list of dictionaries to load the data from.

    Runtime parameters:
        - `batch_size`: The batch size to use when processing the data.

    Output columns:
        - dynamic (based on the keys found on the first dictionary of the list): The columns
            of the dataset.

    Categories:
        - load

    Examples:
        Load data from a list of dictionaries:

        ```python
        from distilabel.steps import LoadDataFromDicts

        loader = LoadDataFromDicts(
            data=[{"instruction": "What are 2+2?"}] * 5,
            batch_size=2
        )
        loader.load()

        result = next(loader.process())
        # >>> result
        # ([{'instruction': 'What are 2+2?'}, {'instruction': 'What are 2+2?'}], False)
        ```
    """

    data: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)

    @override
    def process(self, offset: int = 0) -> "GeneratorStepOutput":  # type: ignore
        """Yields batches from a list of dictionaries.

        Args:
            offset: The offset to start the generation from. Defaults to `0`.
                Processing does not consume or modify the configured data, so offsets
                always refer to the original input on repeated calls.

        Yields:
            A list of Python dictionaries as read from the inputs (propagated in batches)
            and a flag indicating whether the yield batch is the last one.
        """
        # Copy only the current batch, not the entire remaining input each time.
        # Keeping data intact also makes repeated runs and resume offsets stable.
        length = len(self.data)
        first = max(length + offset, 0) if offset < 0 else offset
        for start in range(first, length, self.batch_size):
            end = start + self.batch_size
            yield self.data[start:end], end >= length

    @property
    def outputs(self) -> List[str]:
        """Returns a list of strings with the names of the columns that the step will generate."""
        return list(self.data[0].keys())
