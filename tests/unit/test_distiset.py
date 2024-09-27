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

import copy
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import yaml
from datasets import Dataset, DatasetDict
from upath import UPath

from distilabel.distiset import Distiset


@pytest.fixture(scope="function")
def distiset():
    return Distiset(
        {
            "leaf_step_1": Dataset.from_dict({"a": [1, 2, 3]}),
            "leaf_step_2": Dataset.from_dict({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}),
        }
    )


def make_fake_file(filename: Path) -> None:
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    filename.touch()


def add_config_to_distiset(distiset: Distiset, folder: Path) -> Distiset:
    from distilabel.distiset import DISTISET_CONFIG_FOLDER

    pipeline_yaml = folder / DISTISET_CONFIG_FOLDER / "pipeline.yaml"
    pipeline_log = folder / DISTISET_CONFIG_FOLDER / "pipeline.log"
    make_fake_file(pipeline_yaml)
    make_fake_file(pipeline_log)
    distiset.pipeline_path = pipeline_yaml
    distiset.pipeline_log_path = pipeline_log
    return distiset


class TestDistiset:
    def test_train_test_split(self, distiset: Distiset) -> None:
        assert isinstance(distiset["leaf_step_1"], Dataset)
        ds = distiset.train_test_split(0.8)
        assert isinstance(ds, Distiset)
        assert len(ds) == 2
        assert isinstance(ds["leaf_step_1"], DatasetDict)

    @pytest.mark.parametrize("storage_options", [None, {"test": "option"}])
    @pytest.mark.parametrize("with_config", [False, True])
    def test_save_to_disk(
        self,
        distiset: Distiset,
        with_config: bool,
        storage_options: Optional[Dict[str, Any]],
    ) -> None:
        full_distiset = copy.deepcopy(distiset)
        # Distiset with Distiset
        with tempfile.TemporaryDirectory() as tmpdirname:
            folder = Path(tmpdirname) / "distiset_folder"
            if with_config:
                full_distiset = add_config_to_distiset(full_distiset, folder)

            full_distiset.save_to_disk(
                folder,
                save_card=with_config,
                save_pipeline_config=with_config,
                save_pipeline_log=with_config,
                storage_options=storage_options,
            )
            assert folder.is_dir()
            assert len(list(folder.iterdir())) == 3

        full_distiset = copy.deepcopy(distiset)
        # Distiset with DatasetDict
        distiset_with_dict = full_distiset.train_test_split(0.8)
        with tempfile.TemporaryDirectory() as tmpdirname:
            folder = Path(tmpdirname) / "distiset_folder"
            if with_config:
                distiset_with_dict = add_config_to_distiset(distiset_with_dict, folder)

            distiset_with_dict.save_to_disk(
                folder,
                save_card=with_config,
                save_pipeline_config=with_config,
                save_pipeline_log=with_config,
            )

            assert folder.is_dir()
            assert len(list(folder.iterdir())) == 3

    @pytest.mark.parametrize("pathlib_implementation", [Path, UPath])
    @pytest.mark.parametrize("storage_options", [None, {"project": "experiments"}])
    @pytest.mark.parametrize("with_config", [False, True])
    def test_load_from_disk(
        self,
        distiset: Distiset,
        with_config: bool,
        storage_options: Optional[Dict[str, Any]],
        pathlib_implementation: type,
    ) -> None:
        full_distiset = copy.deepcopy(distiset)
        # Distiset with Distiset
        with tempfile.TemporaryDirectory() as tmpdirname:
            # This way we can test also we work with UPath, using FilePath protocol, as it should
            # do the same as S3Path, GCSPath, etc.
            folder = pathlib_implementation(tmpdirname) / "distiset_folder"
            if with_config:
                full_distiset = add_config_to_distiset(full_distiset, folder)
            full_distiset.save_to_disk(
                folder,
                save_card=with_config,
                save_pipeline_config=with_config,
                save_pipeline_log=with_config,
                storage_options=storage_options,
            )
            ds = Distiset.load_from_disk(
                folder,
                storage_options=storage_options,
            )
            assert isinstance(ds, Distiset)
            assert isinstance(ds["leaf_step_1"], Dataset)

            if with_config:
                assert ds.pipeline_path.exists()
                assert ds.log_filename_path.exists()

        full_distiset = copy.deepcopy(distiset)
        # Distiset with DatasetDict
        distiset_with_dict = full_distiset.train_test_split(0.8)
        with tempfile.TemporaryDirectory() as tmpdirname:
            folder = pathlib_implementation(tmpdirname) / "distiset_folder"
            if with_config:
                distiset_with_dict = add_config_to_distiset(distiset_with_dict, folder)

            distiset_with_dict.save_to_disk(folder)
            ds = Distiset.load_from_disk(folder, storage_options=storage_options)

            assert folder.is_dir()
            assert isinstance(ds["leaf_step_1"], DatasetDict)

            if with_config:
                assert ds.pipeline_path.exists()
                assert ds.log_filename_path.exists()

    def test_dataset_card(self, distiset: Distiset) -> None:
        # Test the the metadata we generate by default without extracting the already generated content from the HF hub.
        # We parse the content and check it's the same as the one we generate.
        distiset_card = distiset._get_card("repo_name_or_path")
        metadata = re.findall(r"---\n(.*?)\n---", str(distiset_card), re.DOTALL)[0]
        metadata = yaml.safe_load(metadata)
        assert metadata == {
            "size_categories": "n<1K",
            "tags": ["synthetic", "distilabel", "rlaif"],
        }
