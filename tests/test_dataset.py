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
from distilabel.dataset import CustomDataset
from distilabel.tasks import UltraFeedbackTask
from distilabel.utils.dataset import DatasetCheckpoint


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


def test_do_checkpoint():
    chk = DatasetCheckpoint(save_frequency=2)
    assert chk.do_checkpoint(0) is False
    assert chk._total_checks == 0
    assert chk.do_checkpoint(2) is True
    assert chk._total_checks == 1
    assert chk.do_checkpoint(3) is False
    assert chk._total_checks == 1
    assert chk.do_checkpoint(4) is True
    assert chk._total_checks == 2
