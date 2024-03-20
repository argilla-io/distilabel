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
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from pyarrow.lib import ArrowInvalid

from distilabel.utils.logging import get_logger

logger = get_logger("utils.data")


class Distiset(dict):
    """Convenient wrapper around `datasets.Dataset` to push to the Hugging Face Hub.

    It's a dictionary where the keys correspond to the different leaf_steps from the internal
    `DAG` and the values are `datasets.Dataset`.
    """

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: Optional[str] = None,
        private: bool = False,
        token: Optional[str] = None,
    ) -> None:
        """Pushes the `Distiset` to the Hugging Face Hub, each dataset will be pushed as a different configuration
        corresponding to the leaf step that generated it.

        Args:
            repo_id:
                The ID of the repository to push to in the following format: `<user>/<dataset_name>` or
                `<org>/<dataset_name>`. Also accepts `<dataset_name>`, which will default to the namespace
                of the logged-in user.
            commit_message: Message to commit while pushing. Will default to `"Upload dataset"`.
            private:
                Whether the dataset repository should be set to private or not. Only affects repository creation:
                a repository that already exists will not be affected by that parameter.
            token:
                An optional authentication token for the Hugging Face Hub. If no token is passed, will default
                to the token saved locally when logging in with `huggingface-cli login`. Will raise an error
                if no token is passed and the user is not logged-in.
        """
        for name, dataset in self.items():
            dataset.push_to_hub(
                repo_id=repo_id,
                config_name=name,
                commit_message=commit_message,
                private=private,
                token=token,
            )

    def train_test_split(
        self,
        train_size: float,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> "Distiset":
        """Return a `Distiset` whose values will be a `datasets.DatasetDict` with two random train and test subsets.
        Splits are created from the dataset according to `train_size` and `shuffle`.

        Args:
            train_size:
                Float between `0.0` and `1.0` representing the proportion of the dataset to include in the test split.
                It will be applied to all the datasets in the `Distiset`.
            shuffle: Whether or not to shuffle the data before splitting
            seed:
                A seed to initialize the default BitGenerator, passed to the underlying method.

        Returns:
            The `Distiset` with the train-test split applied to all the datasets.
        """
        assert 0 < train_size < 1, "train_size must be a float between 0 and 1"
        for name, dataset in self.items():
            self[name] = dataset.train_test_split(
                train_size=train_size,
                shuffle=shuffle,
                seed=seed,
            )
        return self

    def __repr__(self):
        # Copy from `datasets.DatasetDict.__repr__`.
        repr = "\n".join([f"{k}: {v}" for k, v in self.items()])
        repr = re.sub(r"^", " " * 4, repr, count=0, flags=re.M)
        return f"Distiset({{\n{repr}\n}})"


def _create_dataset(data_dir: Path) -> Distiset:
    """Creates a `Distiset` from the buffer folder.

    Args:
        data_dir: Folder where the data buffers were written by the `_WriteBuffer`.
            It should correspond to `CacheLocation.data`.

    Returns:
        The dataset created from the buffer folder, where the different leaf steps will
        correspond to different configurations of the dataset.
    """
    distiset = Distiset()
    for file in data_dir.iterdir():
        if file.suffix != ".parquet":
            continue
        try:
            distiset[file.stem] = load_dataset(
                "parquet", name=file.stem, data_files={"train": str(file)}
            )["train"]
        except ArrowInvalid:
            logger.warning(f"❌ Failed to load the subset from {file}.")
            continue

    if len(distiset) == 0:
        logger.warning(
            "❌ No parquet files found in the buffer. Returning an empty dataset."
        )
    return distiset
