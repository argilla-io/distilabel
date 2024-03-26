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

from pathlib import Path
from typing import List


def list_files_in_dir(dir_path: Path) -> List[Path]:
    """List all files in a directory.

    Args:
        dir_path: Path to the directory.

    Returns:
        A list of file names in the directory.
    """
    return [f for f in dir_path.iterdir() if f.is_file()]
