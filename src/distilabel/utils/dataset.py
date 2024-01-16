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

import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

import dill as pickle

from distilabel.tasks.preference.base import PreferenceTask

if TYPE_CHECKING:
    from distilabel.dataset import CustomDataset
    from distilabel.tasks.base import Task


TASK_FILE_NAME = "task.pkl"


BinarizationStrategies = Literal["random", "worst"]


def save_task_to_disk(path: Path, task: "Task") -> None:
    """Saves a task to disk.

    Args:
        path: The path to the task.
        task: The task.
    """
    task_path = path / TASK_FILE_NAME
    with open(task_path, "wb") as f:
        f.write(pickle.dumps(task))


def load_task_from_disk(path: Path) -> "Task":
    """Loads a task from disk.

    Args:
        path: The path to the task.

    Returns:
        Task: The task.
    """
    task_path = path / "task.pkl"
    if not task_path.exists():
        raise FileNotFoundError(f"The task file does not exist: {task_path}")
    with open(task_path, "rb") as f:
        task = pickle.loads(f.read())
    return task


def _binarize_dataset(
    dataset: "CustomDataset",
    seed: int = 42,
    strategy: BinarizationStrategies = "random",
) -> "CustomDataset":
    rating_column = "rating"
    responses_column = "generations"

    def binarize_random(example):
        random.seed(seed)

        prompt = example["input"]
        best_rating = max(example[rating_column])
        best_response_idx = example[rating_column].index(best_rating)
        chosen_response = example[responses_column][best_response_idx]
        chosen_model = example["generation_model"][best_response_idx]

        # Remove best response
        example[rating_column].pop(best_response_idx)
        example[responses_column].pop(best_response_idx)
        example["generation_model"].pop(best_response_idx)

        # Select the random response
        random_response = random.choice(example[responses_column])
        random_response_idx = example[responses_column].index(random_response)
        random_rating = example[rating_column][random_response_idx]
        random_model = example["generation_model"][random_response_idx]

        binarized = {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": random_response,
            "rating_chosen": int(best_rating),
            "rating_rejected": int(random_rating),
            "chosen_model": chosen_model,
            "rejected_model": random_model,
        }
        return binarized

    def binarize_worst(example):
        random.seed(seed)

        prompt = example["input"]
        best_rating = max(example[rating_column])
        best_response_idx = example[rating_column].index(best_rating)
        chosen_response = example[responses_column][best_response_idx]
        chosen_model = example["generation_model"][best_response_idx]

        worst_rating = min(example[rating_column])
        worst_response_idx = example[rating_column].index(worst_rating)
        worst_response = example[responses_column][worst_response_idx]
        worst_model = example["generation_model"][worst_response_idx]

        binarized = {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": worst_response,
            "rating_chosen": int(best_rating),
            "rating_rejected": int(worst_rating),
            "chosen_model": chosen_model,
            "rejected_model": worst_model,
        }
        return binarized

    if strategy == "random":
        binarization_method = binarize_random
    elif strategy == "worst":
        binarization_method = binarize_worst
    else:
        raise ValueError(
            f"Strategy `{strategy}` is not implemented, it must be one of: {get_args(BinarizationStrategies)}"
        )

    return dataset.map(binarization_method).filter(
        lambda example: example["rating_chosen"] != example["rating_rejected"]
    )


def prepare_dataset(
    dataset: "CustomDataset",
    strategy: BinarizationStrategies = "random",
    seed: int = 42,
) -> "CustomDataset":
    """Helper function to prepare a dataset for training assuming the standard formats.

    Expected format for a dataset to be trained with DPO as defined in trl's
    [dpo trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer#expected-dataset-format).

    Args:
        dataset (CustomDataset): Dataset with a PreferenceTask.
        strategy (BinarizationStrategies, optional):
            Strategy to binarize the data. Defaults to "random".

    Returns:
        CustomDataset: Dataset formatted for training with DPO.
    """
    if not isinstance(dataset.task, PreferenceTask):
        raise ValueError(
            "This functionality is currently implemented for `PreferenceTask` only."
        )

    remove_columns = [
        "input",
        "generation_model",
        "generations",
        "rating",
        "labelling_model",
        "labelling_prompt",
        "raw_labelling_response",
        "rationale",
    ]

    ds = _binarize_dataset(dataset, strategy=strategy, seed=42)

    from distilabel.dataset import CustomDataset

    ds = ds.remove_columns(remove_columns)
    ds.__class__ = CustomDataset
    ds.task = dataset.task
    return ds
