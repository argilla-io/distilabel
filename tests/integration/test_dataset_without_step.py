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

from distilabel.pipeline import Pipeline
from distilabel.steps import make_generator_step
from distilabel.steps.base import Step, StepInput
from distilabel.steps.typing import StepColumns, StepOutput


class DummyStep(Step):
    inputs: StepColumns = ["instruction"]
    outputs: StepColumns = ["response"]

    def process(self, inputs: StepInput) -> StepOutput:  # type: ignore
        for input in inputs:
            input["response"] = "unit test"
        yield inputs


data = [{"instruction": "Tell me a joke."}] * 10


@pytest.mark.parametrize("dataset", (data, Dataset.from_list(data), pd.DataFrame(data)))
def test_pipeline_with_dataset_from_function(
    dataset: Union[Dataset, pd.DataFrame, List[Dict[str, str]]],
) -> None:
    with Pipeline(name="pipe-nothing") as pipeline:
        load_dataset = make_generator_step(dataset)
        if isinstance(dataset, (pd.DataFrame, Dataset)):
            assert isinstance(load_dataset._dataset, Dataset)

        dummy = DummyStep()
        load_dataset >> dummy

    distiset = pipeline.run(use_cache=False)
    assert len(distiset["default"]["train"]) == 10


@pytest.mark.parametrize("dataset", (data, Dataset.from_list(data), pd.DataFrame(data)))
def test_pipeline_run_without_generator_step(
    dataset: Union[Dataset, pd.DataFrame, List[Dict[str, str]]],
) -> None:
    with Pipeline(name="pipe-nothing") as pipeline:
        DummyStep()
        assert len(pipeline.dag) == 1

    distiset = pipeline.run(use_cache=False, dataset=dataset)
    assert len(distiset["default"]["train"]) == 10
    assert len(pipeline.dag) == 2


if __name__ == "__main__":
    with Pipeline(name="pipe-nothing") as pipeline:
        data = [{"instruction": "Tell me a joke."}] * 10
        load_dataset = make_generator_step(Dataset.from_list(data))

        dummy = DummyStep()
        load_dataset >> dummy

    distiset = pipeline.run(use_cache=False)
    print(distiset)
