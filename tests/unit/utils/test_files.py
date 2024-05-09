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

from distilabel.utils.files import list_files_in_dir


def test_list_files_in_dir() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        created_files = []
        for i in range(10):
            file_path = temp_dir / f"{i}.txt"
            created_files.append(file_path)
            with open(file_path, "w") as f:
                f.write("hello")

        assert list_files_in_dir(Path(temp_dir)) == created_files
