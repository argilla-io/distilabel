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

import os
from typing import Generator, Union

import pytest
from datasets import Dataset, IterableDataset
from distilabel.pipeline import Pipeline
from distilabel.steps.generators.huggingface import LoadHubDataset

DISTILABEL_RUN_SLOW_TESTS = os.getenv("DISTILABEL_RUN_SLOW_TESTS", False)


@pytest.fixture(scope="module")
def dataset_loader() -> Generator[Union[Dataset, IterableDataset], None, None]:
    load_hub_dataset = LoadHubDataset(
        name="load_dataset",
        repo_id="distilabel-internal-testing/instruction-dataset-mini",
        split="test",
        batch_size=2,
        pipeline=Pipeline(name="dataset-pipeline"),
    )
    yield load_hub_dataset


@pytest.mark.skipif(
    not DISTILABEL_RUN_SLOW_TESTS,
    reason="These tests depend on internet connection, are slow and depend mainly on HF API, we don't need to test them often.",
)
class TestLoadHubDataset:
    @pytest.mark.parametrize(
        "streaming, ds_type", [(True, IterableDataset), (False, Dataset)]
    )
    def test_runtime_parameters(self, streaming: bool, ds_type) -> None:
        load_hub_dataset = LoadHubDataset(
            name="load_dataset",
            repo_id="distilabel-internal-testing/instruction-dataset-mini",
            split="test",
            streaming=streaming,
            batch_size=2,
            pipeline=Pipeline(name="dataset-pipeline"),
        )
        load_hub_dataset.load()
        assert isinstance(load_hub_dataset._dataset, ds_type)

        generator_step_output = next(load_hub_dataset.process())
        assert isinstance(generator_step_output, tuple)
        assert isinstance(generator_step_output[1], bool)
        assert len(generator_step_output[0]) == 2

    def test_dataset_outputs(self, dataset_loader: LoadHubDataset) -> None:
        # TODO: This test can be run with/without internet connection, we should emulate it here with a mock.
        assert dataset_loader.outputs == ["prompt", "completion", "meta"]
