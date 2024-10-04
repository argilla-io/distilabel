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

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from upath import UPath

from distilabel.utils.serialization import _Serializable


@dataclass
class _Batch(_Serializable):
    """Dataclass to represent a batch of data to be processed by a `_Step`.

    Attributes:
        seq_no: The sequence number of the batch.
        step_name: The name of the step that will process the batch.
        last_batch: A flag to indicate if the batch is the last one.
        data: The data to be processed.
        data_hash: The hash of the data. Defaults to `None`.
        data_path: The path where the data of the batch is stored. Defaults to `None`.
        accumulated: A flag to indicate if the batch is accumulated.
        created_from: A dictionary containing the `seq_no` of the batches of the steps that
            were used to create this batch.
        size: The size of the batch.
    """

    seq_no: int
    step_name: str
    last_batch: bool
    data: List[List[Dict[str, Any]]] = field(default_factory=list, repr=False)
    data_hash: Optional[str] = None
    data_path: Optional[str] = None
    accumulated: bool = False
    created_from: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    batch_routed_to: List[str] = field(default_factory=list)
    size: int = 0
    _fs: Optional[fsspec.AbstractFileSystem] = None

    def next_batch(self) -> "_Batch":
        """Create a new `_Batch` instance with the next batch of data.

        Args:
            data: The data to be processed.

        Returns:
            A `_Batch` instance.
        """
        return _Batch(
            seq_no=self.seq_no + 1, step_name=self.step_name, last_batch=self.last_batch
        )

    def set_data(self, data: List[List[Dict[str, Any]]]) -> None:
        """Sets the data of the batch and updates the size of the batch.

        Args:
            data: The data of the batch.
        """
        self.data = data
        self.size = len(data[0])
        self._update_data_hash()

    def get_data(self, num_rows: Union[int, None] = None) -> List[Dict[str, Any]]:
        """Takes `num_rows` from the data of the batch and returns it. This method will
        also remove the data from the batch and update the hash of the data.

        Args:
            num_rows: The number of rows to take from the data. If `None`, then all the
                data will be taken. Defaults to `None`.

        Returns:
            A list with the data taken from the batch.
        """

        if self.data == [] and self.data_path is not None:
            pass

        if num_rows is None:
            data = self.data[0]
            self.data = [[]]
        else:
            data = self.data[0][:num_rows]
            self.data[0] = self.data[0][num_rows:]

        self.size = len(self.data[0])
        self._update_data_hash()
        return data

    def _update_data_hash(self) -> None:
        """Updates the hash of the data of the batch."""
        self.data_hash = hashlib.sha1(str(self.data).encode()).hexdigest()

    @classmethod
    def accumulate(cls, step_name: str, batches: List[List["_Batch"]]) -> "_Batch":
        """Creates a `_Batch` instance using the data from the list of batches that
        were received from another steps. The batches will be accumulated in a single
        list of data.

        Args:
            step_name: The name of the step that will process the batch.
            batches: a list containing the list of batches received from the predecessors.

        Returns:
            A `_Batch` instance.
        """

        data = []
        for step_batches in batches:
            accumulated_data = [row for batch in step_batches for row in batch.data[0]]
            data.append(accumulated_data)
        return cls(
            seq_no=0, step_name=step_name, last_batch=True, data=data, accumulated=True
        )

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_Batch` to a dictionary, using the `dataclass` helper function.

        Args:
            obj: Unused, just kept to match the signature of the parent method.
            kwargs: Additional arguments that are kept to match the signature of the parent method.

        Returns:
            A `dict` containing the internal representation of the `_Batch`.
        """

        include_batch_data = kwargs.get("include_batch_data", True)

        dump = {
            "seq_no": self.seq_no,
            "step_name": self.step_name,
            "last_batch": self.last_batch,
            "data_hash": self.data_hash,
            "accumulated": self.accumulated,
            "created_from": self.created_from,
            "batch_routed_to": self.batch_routed_to,
            "size": self.size,
        }

        if include_batch_data:
            dump["data"] = self.data

        return dump

    def copy(self) -> "_Batch":
        """Creates a copy of the `_Batch` instance.

        Returns:
            A copy of the `_Batch` instance.
        """
        return copy.deepcopy(self)

    def write_batch_data_to_fs(
        self,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        base_path: Optional[UPath] = None,
    ) -> None:
        """Writes the content of the batch to the filesystem.

        Args
            fs: The `fsspec` filesystem to be used to write the data. If not provided, the
                one set in the `_fs` attribute will be used. Defaults to `None`.
            base_path: The base path where the data of the batch will be stored. If not
                provided, the one set in the `data_path` attribute will be used. Defaults
                to `None`.

        Raises:
            ValueError: If `fs` is not provided and the `_fs` attribute is not set.
        """

        if not fs and not self._fs:
            raise ValueError(
                "The `fs` parameter must be provided if the `_fs` attribute is not set."
            )

        if fs:
            self._fs = fs

        if not base_path and not self.data_path:
            raise ValueError(
                "The `base_path` parameter must be provided if the `data_path` attribute"
                " is not set."
            )

        seq_no_dir = (
            base_path / f"seq_no_{self.seq_no}" if base_path else UPath(self.data_path)
        )
        seq_no_dir._fs_cached = self._fs  # type: ignore
        seq_no_dir.mkdir(parents=True, exist_ok=True)

        for i, data in enumerate(self.data):
            table = pa.Table.from_pylist(data)
            with self._fs.open(seq_no_dir / f"data_index_{i}.parquet", "wb") as f:  # type: ignore
                pq.write_table(table, f)

        self.data = []
        self.data_path = str(seq_no_dir)

    def read_batch_data_from_fs(self) -> None:
        """Reads the content of the batch from the filesystem."""
        if not self.data_path:
            raise ValueError(
                "`data_path` attribute must be set to read the data from the filesystem."
                " Use `write_batch_data_to_fs` method to set the `data_path` attribute."
            )

        if not self._fs:
            raise ValueError(
                "`_fs` attribute must be set to read the data from the filesystem."
                " Use `write_batch_data_to_fs` method to set the `_fs` attribute."
            )

        for file in self._fs.ls(self.data_path):
            with self._fs.open(file, "rb") as f:
                table = pq.read_table(f)
                self.data.append(table.to_pylist())

        self._fs.rm(self.data_path, recursive=True)
