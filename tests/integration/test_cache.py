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

from typing import TYPE_CHECKING

import numpy as np
import pytest

from distilabel.pipeline import Pipeline
from distilabel.steps import GeneratorStep, StepInput, step
from distilabel.steps.typing import StepColumns

if TYPE_CHECKING:
    from distilabel.steps import GeneratorStepOutput, StepOutput


class NumpyBigArrayGenerator(GeneratorStep):
    num_batches: int
    outputs: StepColumns = ["array"]

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        for i in range(self.num_batches):
            yield (
                [{"array": np.random.randn(256)} for _ in range(self.batch_size)],  # type: ignore
                i == self.num_batches - 1,
            )  # type: ignore


@step(step_type="global")
def ReceiveArrays(inputs: StepInput) -> "StepOutput":
    yield inputs


@pytest.mark.benchmark
def test_cache_time() -> None:
    with Pipeline(name="dummy") as pipeline:
        numpy_generator = NumpyBigArrayGenerator(num_batches=2, batch_size=100)

        receive_arrays = ReceiveArrays()

        numpy_generator >> receive_arrays

    pipeline.run(use_cache=False)
