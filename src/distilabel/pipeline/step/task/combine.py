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

from abc import ABC, abstractmethod
from typing import Iterator, List

from distilabel.pipeline.step.base import Step
from distilabel.pipeline.step.typing import StepInput
from distilabel.pipeline.utils import combine_dicts


class CombineColumns(Step, ABC):
    """CombineColumns is an abstract Step that implements the `process` method that calls
    the `combine_dicts` function to handle and combine a list of `StepInput`. Anyway, in order
    to use this class, one would still need to implement the `inputs` and `outputs` properties
    that will be the columns to merge and the name of those merged columns."""

    @property
    @abstractmethod
    def inputs(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def outputs(self) -> List[str]:
        ...

    def process(self, *args: StepInput) -> Iterator[StepInput]:
        yield combine_dicts(
            *args,
            merge_keys=set(self.inputs),
            output_merge_keys=set(self.outputs),
        )
