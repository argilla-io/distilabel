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

from datasets import DatasetDict, load_dataset
from pyarrow.lib import ArrowInvalid

from distilabel.utils.logging import get_logger

logger = get_logger("data")


def _create_dataset(data_dir: Path) -> DatasetDict:
    """Creates a datasets.Dataset from the buffer folder.

    Args:
        data_dir (Path): Folder where the data buffers were written by the _WriteBuffer.
            It should correspond to `CacheLocation.data`.

    Returns:
        datasets.DatasetDict: The dataset created from the buffer folder,
            where the different leaf steps will correspond to different configurations
            of the dataset.
    """
    data_files = {}
    for file in data_dir.iterdir():
        if file.suffix != ".parquet":
            continue
        data_files[file.stem] = str(file)

    if len(data_files) == 0:
        logger.warning(
            "❌ No parquet files found in the buffer. Returning an empty dataset."
        )
        return DatasetDict()

    try:
        return load_dataset("parquet", data_files=data_files)
    except ArrowInvalid:
        logger.warning(
            "❌ Failed to load the dataset from the buffer. Returning an empty dataset."
        )
        return DatasetDict()
