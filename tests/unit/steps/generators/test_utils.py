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

from typing import Dict, List, Union

import pandas as pd
import pytest
from datasets import Dataset

from distilabel.pipeline.local import Pipeline
from distilabel.steps.generators.utils import make_generator_step

data = [{"instruction": "Tell me a joke."}] * 10


@pytest.mark.parametrize("dataset", (data, Dataset.from_list(data), pd.DataFrame(data)))
def test_make_generator_step(
    dataset: Union[Dataset, pd.DataFrame, List[Dict[str, str]]],
) -> None:
    batch_size = 5
    load_dataset = make_generator_step(
        dataset, batch_size=batch_size, output_mappings={"instruction": "other"}
    )
    load_dataset.load()
    result = next(load_dataset.process())
    assert len(result[0]) == batch_size
    if isinstance(dataset, (pd.DataFrame, Dataset)):
        assert isinstance(load_dataset._dataset, Dataset)
    else:
        assert isinstance(load_dataset.data, list)

    assert load_dataset.output_mappings == {"instruction": "other"}


def test_make_generator_step_with_pipeline() -> None:
    pipeline = Pipeline()
    load_dataset = make_generator_step(data, pipeline=pipeline)
    assert load_dataset.pipeline == pipeline
