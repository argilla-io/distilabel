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

import logging
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Set

import pyarrow as pa
import pyarrow.parquet as pq

from distilabel.pipeline.batch import _Batch
from distilabel.utils.dicts import flatten_dict
from distilabel.utils.files import list_files_in_dir


class _WriteBuffer:
    """Class in charge of sending the batched contents to a buffer and writing
    those to files under a given folder.

    As batches are received, they are added to the buffer and once each buffer
    is full, the content is written to a parquet file.
    """

    def __init__(self, path: "PathLike", leaf_steps: Set[str]) -> None:
        """
        Args:
            path: Folder where the files will be written, the idea
                is for this path to be in the cache folder under /data.
            leaf_steps: Leaf steps from either the DAG of the Pipeline.

        Raises:
            ValueError: If the path is not a directory.
        """
        self._path = Path(path)
        if not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)
            for step in leaf_steps:
                (self._path / step).mkdir(parents=True, exist_ok=True)

        if not self._path.is_dir():
            raise ValueError(f"The path should be a directory, not a file: {path}")

        self._buffers: Dict[str, List[Dict[str, Any]]] = {
            step: [] for step in leaf_steps
        }
        # TODO: make this configurable
        self._buffers_dump_batch_size: Dict[str, int] = {
            step: 50 for step in leaf_steps
        }
        self._buffer_last_schema = {}
        self._buffers_last_file: Dict[str, int] = {step: 1 for step in leaf_steps}
        self._logger = logging.getLogger("distilabel.write_buffer")

    def _get_filename(self, step_name: str) -> Path:
        """Creates the filename for the step.

        Args:
            step_name: Name of the step to which the data belongs to.

        Returns:
            Filename for the step.
        """
        return self._path / f"{step_name}.parquet"

    def is_full(self, step_name: str) -> bool:
        """Checks the buffers that are full so that those can be written to the file.

        Returns:
            Whether the buffer is full.
        """
        return len(self._buffers[step_name]) >= self._buffers_dump_batch_size[step_name]

    def add_batch(self, batch: "_Batch") -> None:
        """Adds a batch to the buffer and writes the buffer to the file if it's full.

        Args:
            batch: batch to add to the buffer.
        """
        step_name = batch.step_name
        data = batch.data[0]
        self._buffers[step_name].extend(data)
        self._logger.debug(
            f"Added batch to write buffer for step '{step_name}' with {len(data)} rows."
        )
        if self.is_full(step_name):
            self._logger.debug(
                f"Buffer for step '{step_name}' is full (rows: {len(self._buffers[step_name])},"
                f" full: {self._buffers_dump_batch_size[step_name]}), writing to file..."
            )
            self._write(step_name)

    def _write(self, step_name: str) -> None:
        """Writes the content to the file and cleans the buffer.

        Args:
            step_name (str): Name of the step to which the data pertains.
        """
        step_parquet_dir = Path(self._path, step_name)
        if not step_parquet_dir.exists():
            self._logger.debug(
                f"Creating directory for step '{step_name}' parquet files..."
            )
            step_parquet_dir.mkdir()

        try:
            table = pa.Table.from_pylist(self._buffers[step_name])
        except pa.lib.ArrowInvalid as pae:
            if (
                repr(pae)
                != "ArrowInvalid('cannot mix struct and non-struct, non-null values')"
            ):
                raise pae
            flattened_buffers = [flatten_dict(buf) for buf in self._buffers[step_name]]
            table = pa.Table.from_pylist(flattened_buffers)

        last_schema = self._buffer_last_schema.get(step_name)
        if last_schema is None:
            self._buffer_last_schema[step_name] = table.schema
        else:
            if not last_schema.equals(table.schema):
                new_schema = pa.unify_schemas([last_schema, table.schema])
                self._buffer_last_schema[step_name] = new_schema
                table = table.cast(new_schema)

        next_file_number = self._buffers_last_file[step_name]
        self._buffers_last_file[step_name] = next_file_number + 1

        parquet_file = step_parquet_dir / f"{str(next_file_number).zfill(5)}.parquet"
        pq.write_table(table, parquet_file)
        self._logger.debug(f"Written to file '{parquet_file}'")

        self._clean_buffer(step_name)

    def _clean_buffer(self, step_name: str) -> None:
        """Cleans the buffer by setting it's content to `None`.

        Args:
            step_name: The name of the buffer to clean.
        """
        self._buffers[step_name] = []

    def close(self) -> None:
        """Closes the buffer by writing the remaining content to the file."""
        for step_name in self._buffers:
            if self._buffers[step_name]:
                self._write(step_name)

            # We need to read the parquet files and write them again to ensure the schema
            # is correct. Otherwise, the first parquets won't have the last schema and
            # then we will have issues when reading them.
            for file in list_files_in_dir(self._path / step_name):
                if step_name in self._buffer_last_schema:
                    table = pq.read_table(
                        file, schema=self._buffer_last_schema[step_name]
                    )
                    pq.write_table(table, file)
