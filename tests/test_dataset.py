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

import tempfile
from pathlib import Path

import pytest
from distilabel.dataset import CustomDataset, DatasetCheckpoint
from distilabel.tasks import UltraFeedbackTask


@pytest.fixture
def custom_dataset():
    ds = CustomDataset.from_dict({"input": ["a", "b"], "generations": ["c", "d"]})
    ds.task = UltraFeedbackTask.for_text_quality()
    return ds


def test_dataset_save_to_disk(custom_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_name = Path(tmpdir) / "dataset_folder"
        custom_dataset.save_to_disk(ds_name)
        assert ds_name.is_dir()
        assert (ds_name / "task.pkl").is_file()


def test_dataset_load_disk(custom_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_name = Path(tmpdir) / "dataset_folder"
        custom_dataset.save_to_disk(ds_name)
        ds_from_disk = CustomDataset.load_from_disk(ds_name)
        assert isinstance(ds_from_disk, CustomDataset)
        assert isinstance(ds_from_disk.task, UltraFeedbackTask)


@pytest.mark.parametrize(
    "save_frequency, dataset_len, batch_size, expected",
    [
        (1, 10, 1, 10),
        (3, 10, 1, 3),
        (8, 32, 8, 4),
        (8, 64, 16, 4),
        (20, 100, 7, 5),
    ],
)
def test_do_checkpoint(
    save_frequency: int, dataset_len: int, batch_size: int, expected: int
):
    ds = CustomDataset.from_dict(
        {"input": ["a"] * dataset_len, "generations": ["a"] * dataset_len}
    )
    chk = DatasetCheckpoint(save_frequency=save_frequency)
    ctr = 0
    for batch_i, _ in enumerate(ds.iter(batch_size=batch_size), start=1):
        step = batch_i * batch_size
        if chk.do_checkpoint(step):
            ctr += 1
    assert ctr == expected == chk._total_checks
