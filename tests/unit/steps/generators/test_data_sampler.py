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

from typing import List

import pytest

from distilabel.steps.generators.data_sampler import DataSampler


@pytest.mark.parametrize(
    "samples, size, batch_size, expected",
    [
        (10, 2, 4, [4, 4, 2]),
        (7, 5, 6, [6, 1]),
        (20, 5, 20, [20]),
        (20, 50, 8, [8, 8, 4]),
    ],
)
def test_generator_and_sampler(
    samples: int, size: int, batch_size: int, expected: List[int]
):
    sampler = DataSampler(
        data=[{"sample": f"sample {i}"} for i in range(30)],
        size=size,
        samples=samples,
        batch_size=batch_size,
    )
    sampler.load()
    results = [item[0] for item in sampler.process()]
    assert len(results) == len(expected)
    assert len(results[0]) == batch_size
    for i, result in enumerate(results):
        assert len(result) == expected[i]
