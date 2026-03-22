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

import shutil
from pathlib import Path
from typing import Optional, Tuple


def find_latest_snapshot(snapshots_path: Path) -> Optional[Path]:
    """Find the most recent snapshot directory by modification time.

    Args:
        snapshots_path: The directory containing snapshot subdirectories.

    Returns:
        The path to the most recent snapshot directory, or None if no snapshots exist.
    """
    if not snapshots_path.exists():
        return None

    snapshot_dirs = [d for d in snapshots_path.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        return None

    return max(snapshot_dirs, key=lambda d: d.stat().st_mtime)


def find_cache_root(input_path: Path, pipeline_name: str) -> Optional[Path]:
    """Navigate from input_path to find the cache root containing steps_data/.

    Looks for a steps_data directory either directly under input_path or by
    traversing up the directory tree looking for a directory that contains
    steps_data as a subdirectory.

    Args:
        input_path: The starting path to search from.
        pipeline_name: The name of the pipeline (used for path resolution).

    Returns:
        The path to the directory containing steps_data/, or None if not found.
    """
    # Direct check: input_path/steps_data/
    steps_data = input_path / "steps_data"
    if steps_data.exists() and steps_data.is_dir():
        return input_path

    # Check parent directories up to 3 levels
    current = input_path
    for _ in range(3):
        parent = current.parent
        if parent == current:
            break
        steps_data = parent / "steps_data"
        if steps_data.exists() and steps_data.is_dir():
            return parent
        current = parent

    return None


def copy_directory_with_stats(src: Path, dst: Path) -> Tuple[int, int]:
    """Copy directory tree, return (files_copied, bytes_copied).

    Args:
        src: Source directory to copy from.
        dst: Destination directory to copy to.

    Returns:
        A tuple of (files_copied, bytes_copied).
    """
    files_copied = 0
    bytes_copied = 0

    if not src.exists():
        return files_copied, bytes_copied

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.rglob("*"):
        if item.is_file():
            relative = item.relative_to(src)
            dest_file = dst / relative
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(dest_file))
            files_copied += 1
            bytes_copied += item.stat().st_size

    return files_copied, bytes_copied


def get_directory_stats(path: Path) -> Tuple[int, int]:
    """Return (file_count, total_bytes) for a directory.

    Args:
        path: The directory to get stats for.

    Returns:
        A tuple of (file_count, total_bytes).
    """
    if not path.exists():
        return 0, 0

    file_count = 0
    total_bytes = 0

    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size

    return file_count, total_bytes
