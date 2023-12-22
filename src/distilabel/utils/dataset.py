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

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from distilabel.tasks.base import Task


def save_task_to_disk(path: Path, task: "Task") -> None:
    """Saves a task to disk.

    Args:
        path: The path to the task.
        task: The task.
    """
    task_path = path / "task.pkl"
    with open(task_path, "wb") as f:
        pickle.dump(task, f)


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
        task = pickle.load(f)
    return task


@dataclass
class DatasetCheckpoint:
    """A checkpoint class that contains the information of a checkpoint.

    Args:
        path (Path): The path to the checkpoint.
        save_frequency (int): The frequency at which the checkpoint should be saved
            By default is set to -1 (no checkpoint is saved to disk, but the dataset
            is returned upon failure).
        extra_kwargs (dict[str, Any]): Additional kwargs to be passed to the `save_to_disk` method of the Dataset.
    """

    path: Path = Path.cwd() / "dataset_checkpoint"
    save_frequency: int = -1
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    # Internal fields to keep track of the number of records generated and when to check.
    _total_checks: int = field(repr=False, default=0)

    def do_checkpoint(self, step: int) -> bool:
        """Determines if a checkpoint should be done.

        Args:
            step (int): The number of records generated.

        Returns:
            bool: Whether a checkpoint should be done.
        """
        if self.save_frequency == -1:
            return False

        if (step - self._total_checks * self.save_frequency) // self.save_frequency:
            self._total_checks += 1
            return True
        return False
