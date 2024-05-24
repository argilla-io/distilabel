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
from typing import Callable, List, Optional


def list_files_in_dir(
    dir_path: Path, key: Optional[Callable] = lambda x: int(x.stem)
) -> List[Path]:
    """List all files in a directory.

    Args:
        dir_path: Path to the directory.
        key: A function to sort the files. Defaults to sorting by the integer value of the file name.
            This is useful when loading files from the cache, as the name will be numbered.

    Returns:
        A list of file names in the directory.
    """
    return [f for f in sorted(dir_path.iterdir(), key=key) if f.is_file()]
