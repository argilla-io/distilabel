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
from pathlib import Path
from typing import TYPE_CHECKING

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
