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

import re
import tempfile
from pathlib import Path
from typing import List

import pytest
from distilabel.dataset import CustomDataset, DatasetCheckpoint
from distilabel.tasks import TextGenerationTask, UltraFeedbackTask
from distilabel.utils.dataset import prepare_dataset


@pytest.fixture
def custom_dataset():
    ds = CustomDataset.from_dict({"input": ["a", "b"], "generations": ["c", "d"]})
    ds.task = UltraFeedbackTask.for_overall_quality()
    return ds


@pytest.fixture
def sample_preference_dataset():
    ds = CustomDataset.from_dict(
        {
            "input": ["input 1", "input 2", "input 3"],
            "generation_model": [
                [
                    "argilla/notus-7b-v1",
                    "WizardLM/WizardCoder-15B-V1.0",
                    "ise-uiuc/Magicoder-S-DS-6.7B",
                    "gpt-3.5-turbo",
                ],
                [
                    "argilla/notus-7b-v1",
                    "ise-uiuc/Magicoder-S-DS-6.7B",
                    "WizardLM/WizardCoder-15B-V1.0",
                    "gpt-3.5-turbo",
                ],
                [
                    "argilla/notus-7b-v1",
                    "ise-uiuc/Magicoder-S-DS-6.7B",
                    "WizardLM/WizardCoder-15B-V1.0",
                    "gpt-3.5-turbo",
                ],
            ],
            "generations": [
                [
                    "generation 1 1",
                    "generation 1 2",
                    "generation 1 3",
                    "generation 1 4",
                ],
                [
                    "generation 2 1",
                    "generation 2 2",
                    "generation 2 3",
                    "generation 2 4",
                ],
                [
                    "generation 3 1",
                    "generation 3 2",
                    "generation 3 3",
                    "generation 3 4",
                ],
            ],
            "labelling_model": [
                "gpt-4-1106-preview",
                "gpt-4-1106-preview",
                "gpt-4-1106-preview",
            ],
            "labelling_prompt": [
                [
                    {
                        "content": "Your role is to evaluate text quality based on given criteria.",
                        "role": "system",
                    },
                    {"content": "content", "role": "user"},
                ],
                [
                    {
                        "content": "Your role is to evaluate text quality based on given criteria.",
                        "role": "system",
                    },
                    {"content": "content", "role": "user"},
                ],
                [
                    {
                        "content": "Your role is to evaluate text quality based on given criteria.",
                        "role": "system",
                    },
                    {"content": "content", "role": "user"},
                ],
            ],
            "raw_labelling_response": ["response", "response", "response"],
            "rating": [
                [2.0, 5.0, 4.0, 5.0],
                [2.0, 3.0, 1.0, 4.0],
                [4.0, 3.0, 5.0, 3.0],
            ],
            "rationale": [
                ["rationale 1", "rationale 2", "rationale 3", "rationale 4"],
                ["rationale 1", "rationale 2", "rationale 3", "rationale 4"],
                ["rationale 1", "rationale 2", "rationale 3", "rationale 4"],
            ],
        }
    )
    ds.task = UltraFeedbackTask.for_overall_quality()
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
    ds.task = UltraFeedbackTask.for_overall_quality()
    chk = DatasetCheckpoint(save_frequency=save_frequency)
    ctr = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_name = Path(tmpdir) / "dataset_folder"
        for batch_i, _ in enumerate(ds.iter(batch_size=batch_size), start=1):
            step = batch_i * batch_size
            if chk.do_checkpoint(step):
                ds.save_to_disk(ds_name)
                ds_from_disk = CustomDataset.load_from_disk(ds_name)
                assert ds_from_disk.to_pandas()["generations"].isna().sum() == 0

                ctr += 1
    assert ctr == expected == chk._total_checks


@pytest.mark.parametrize(
    "with_generation_model",
    [True],
)
@pytest.mark.parametrize(
    "strategy, chosen, rejected, chosen_model, rejected_model, keep_ties",
    [
        (
            "random",
            [
                [
                    {"content": "input 1", "role": "user"},
                    {"content": "generation 1 2", "role": "assistant"},
                ],
                [
                    {"content": "input 2", "role": "user"},
                    {"content": "generation 2 4", "role": "assistant"},
                ],
                [
                    {"content": "input 3", "role": "user"},
                    {"content": "generation 3 3", "role": "assistant"},
                ],
            ],
            [
                [
                    {"content": "input 1", "role": "user"},
                    {"content": "generation 1 1", "role": "assistant"},
                ],
                [
                    {"content": "input 2", "role": "user"},
                    {"content": "generation 2 3", "role": "assistant"},
                ],
                [
                    {"content": "input 3", "role": "user"},
                    {"content": "generation 3 4", "role": "assistant"},
                ],
            ],
            [
                "WizardLM/WizardCoder-15B-V1.0",
                "gpt-3.5-turbo",
                "WizardLM/WizardCoder-15B-V1.0",
            ],
            ["argilla/notus-7b-v1", "WizardLM/WizardCoder-15B-V1.0", "gpt-3.5-turbo"],
            True,
        ),
        (
            "worst",
            [
                [
                    {"content": "input 1", "role": "user"},
                    {"content": "generation 1 2", "role": "assistant"},
                ],
                [
                    {"content": "input 2", "role": "user"},
                    {"content": "generation 2 4", "role": "assistant"},
                ],
                [
                    {"content": "input 3", "role": "user"},
                    {"content": "generation 3 3", "role": "assistant"},
                ],
            ],
            [
                [
                    {"content": "input 1", "role": "user"},
                    {"content": "generation 1 1", "role": "assistant"},
                ],
                [
                    {"content": "input 2", "role": "user"},
                    {"content": "generation 2 3", "role": "assistant"},
                ],
                [
                    {"content": "input 3", "role": "user"},
                    {"content": "generation 3 2", "role": "assistant"},
                ],
            ],
            [
                "WizardLM/WizardCoder-15B-V1.0",
                "gpt-3.5-turbo",
                "WizardLM/WizardCoder-15B-V1.0",
            ],
            [
                "argilla/notus-7b-v1",
                "WizardLM/WizardCoder-15B-V1.0",
                "ise-uiuc/Magicoder-S-DS-6.7B",
            ],
            True,
        ),
    ],
)
def test_prepare_dataset(
    sample_preference_dataset: CustomDataset,
    strategy: str,
    chosen: List[str],
    rejected: List[str],
    chosen_model: List[str],
    rejected_model: List[str],
    with_generation_model: bool,
    keep_ties: bool,
):
    if not with_generation_model:
        sample_preference_dataset = sample_preference_dataset.remove_columns(
            ["generation_model"]
        )
    ds = prepare_dataset(
        sample_preference_dataset, strategy=strategy, seed=42, keep_ties=keep_ties
    )
    assert isinstance(ds, CustomDataset)
    assert ds.column_names == [
        "prompt",
        "chosen",
        "rejected",
        "rating_chosen",
        "rating_rejected",
        "chosen_model",
        "rejected_model",
    ]
    for i, row in enumerate(ds):
        assert row["chosen"] == chosen[i]
        assert row["rejected"] == rejected[i]
        assert row["chosen_model"] == chosen_model[i]
        assert row["rejected_model"] == rejected_model[i]


def test_prepare_dataset_wrong_task(sample_preference_dataset: CustomDataset):
    sample_preference_dataset.task = TextGenerationTask()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "This functionality is currently implemented for `PreferenceTask` only."
        ),
    ):
        prepare_dataset(sample_preference_dataset)


def test_dataset_wrong_strategy(sample_preference_dataset: CustomDataset):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Strategy `wrong_strategy` is not implemented, it must be one of: ('random', 'worst')"
        ),
    ):
        prepare_dataset(sample_preference_dataset, strategy="wrong_strategy")
