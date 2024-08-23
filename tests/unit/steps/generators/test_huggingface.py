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
import tempfile
from pathlib import Path
from typing import Generator, Union

import pytest
from datasets import Dataset, IterableDataset

from distilabel.distiset import Distiset
from distilabel.pipeline import Pipeline
from distilabel.steps.generators.huggingface import (
    LoadDataFromDisk,
    LoadDataFromFileSystem,
    LoadDataFromHub,
)

DISTILABEL_RUN_SLOW_TESTS = os.getenv("DISTILABEL_RUN_SLOW_TESTS", False)


@pytest.fixture(scope="module")
def dataset_loader() -> Generator[Union[Dataset, IterableDataset], None, None]:
    load_hub_dataset = LoadDataFromHub(
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
class TestLoadDataFromHub:
    @pytest.mark.parametrize(
        "streaming, ds_type", [(True, IterableDataset), (False, Dataset)]
    )
    def test_runtime_parameters(self, streaming: bool, ds_type) -> None:
        load_hub_dataset = LoadDataFromHub(
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

    def test_dataset_outputs(self, dataset_loader: LoadDataFromHub) -> None:
        # TODO: This test can be run with/without internet connection, we should emulate it here with a mock.
        assert dataset_loader.outputs == ["prompt", "completion", "meta"]


class TestLoadDataFromFileSystem:
    @pytest.mark.parametrize("filetype", ["json", None])
    @pytest.mark.parametrize("streaming", [True, False])
    def test_read_from_jsonl(self, streaming: bool, filetype: Union[str, None]) -> None:
        loader = LoadDataFromFileSystem(
            filetype=filetype,
            data_files=str(Path(__file__).parent / "sample_functions.jsonl"),
            streaming=streaming,
        )
        loader.load()
        generator_step_output = next(loader.process())
        assert isinstance(generator_step_output, tuple)
        assert isinstance(generator_step_output[1], bool)
        assert len(generator_step_output[0]) == 11

    @pytest.mark.parametrize("filetype", ["json", None])
    def test_read_from_jsonl_with_folder(self, filetype: Union[str, None]) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "sample_functions.jsonl"
            sample_file = Path(__file__).parent / filename
            for i in range(3):
                Path(tmpdir).mkdir(parents=True, exist_ok=True)
                (Path(tmpdir) / f"sample_functions_{i}.jsonl").write_text(
                    sample_file.read_text(), encoding="utf-8"
                )

            loader = LoadDataFromFileSystem(
                filetype=filetype,
                data_files=tmpdir,
            )
            loader.load()
            generator_step_output = next(loader.process())
            assert isinstance(generator_step_output, tuple)
            assert isinstance(generator_step_output[1], bool)
            assert len(generator_step_output[0]) == 33

    @pytest.mark.parametrize("filetype", ["json", None])
    def test_read_from_jsonl_with_nested_folder(
        self, filetype: Union[str, None]
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "sample_functions.jsonl"
            sample_file = Path(__file__).parent / filename
            for folder in ["train", "validation"]:
                (Path(tmpdir) / folder).mkdir(parents=True, exist_ok=True)
                (Path(tmpdir) / folder / filename).write_text(
                    sample_file.read_text(), encoding="utf-8"
                )

            loader = LoadDataFromFileSystem(
                filetype=filetype,
                data_files=tmpdir,
            )
            loader.load()
            generator_step_output = next(loader.process())
            assert isinstance(generator_step_output, tuple)
            assert isinstance(generator_step_output[1], bool)
            assert len(generator_step_output[0]) == 22

    @pytest.mark.parametrize("load", [True, False])
    def test_outputs(self, load: bool) -> None:
        loader = LoadDataFromFileSystem(
            filetype="json",
            data_files=str(Path(__file__).parent / "sample_functions.jsonl"),
        )
        if load:
            loader.load()
            assert loader.outputs == ["type", "function"]
        else:
            with pytest.raises(ValueError):
                loader.outputs  # noqa: B018


class TestLoadDataFromDisk:
    def test_load_dataset_from_disk(self) -> None:
        dataset = Dataset.from_dict({"a": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = str(Path(tmpdir) / "dataset_path")
            dataset.save_to_disk(dataset_path)

            loader = LoadDataFromDisk(dataset_path=dataset_path)
            loader.load()
            generator_step_output = next(loader.process())
            assert isinstance(generator_step_output, tuple)
            assert isinstance(generator_step_output[1], bool)
            assert len(generator_step_output[0]) == 3

    def test_load_distiset_from_disk(self) -> None:
        distiset = Distiset(
            {
                "leaf_step_1": Dataset.from_dict({"a": [1, 2, 3]}),
                "leaf_step_2": Dataset.from_dict(
                    {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
                ),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = str(Path(tmpdir) / "dataset_path")
            distiset.save_to_disk(dataset_path)

            loader = LoadDataFromDisk(
                dataset_path=dataset_path, is_distiset=True, config="leaf_step_1"
            )
            loader.load()
            generator_step_output = next(loader.process())
            assert isinstance(generator_step_output, tuple)
            assert isinstance(generator_step_output[1], bool)
            assert len(generator_step_output[0]) == 3
