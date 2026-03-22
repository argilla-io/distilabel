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
from typing import TYPE_CHECKING, List

from pydantic import Field, PrivateAttr
from typing_extensions import override

from distilabel.steps.base import GeneratorStep

if TYPE_CHECKING:
    from distilabel.typing import GeneratorStepOutput


class LoadFromCachedOutput(GeneratorStep):
    """Loads data from cached step output batch files (batch_N.json).

    This generator step reads batch files produced by a previous pipeline execution
    and yields the records, allowing pipelines to resume from cached intermediate
    outputs.

    Attributes:
        cache_dir: Path to the {step_name}_{signature}/ directory containing batch files.
        output_columns: The column names to output. If empty, will be auto-detected
            from the first batch file during load().

    Output columns:
        - dynamic (based on the cached data): The columns from the cached step output.

    Categories:
        - load
    """

    cache_dir: str = Field(...)
    output_columns: List[str] = Field(default_factory=list)

    _batch_files: List[Path] = PrivateAttr(default_factory=list)

    def load(self) -> None:
        """Validate directory exists, discover batch_*.json files, auto-detect output_columns."""
        super().load()

        from distilabel.pipeline.batch import _Batch

        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            raise ValueError(f"Cache directory does not exist: {self.cache_dir}")

        self._batch_files = sorted(
            cache_path.glob("batch_*.json"),
            key=lambda f: int(f.stem.split("_")[1]),
        )

        if not self._batch_files:
            raise ValueError(
                f"No batch files found in cache directory: {self.cache_dir}"
            )

        # Auto-detect output columns from first batch if not provided
        if not self.output_columns:
            first_batch = _Batch.from_json(self._batch_files[0])
            if first_batch.data and first_batch.data[0]:
                self.output_columns = list(first_batch.data[0][0].keys())

    @override
    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        """Yield (records, last_batch) from each batch file.

        Args:
            offset: The number of batch files to skip. Defaults to 0.

        Yields:
            A tuple of (list of record dicts, is_last_batch flag).
        """
        from distilabel.pipeline.batch import _Batch

        batch_files = self._batch_files[offset:]
        for i, batch_file in enumerate(batch_files):
            batch = _Batch.from_json(batch_file)
            is_last = i == len(batch_files) - 1
            yield (batch.data[0], is_last)

    @property
    def outputs(self) -> List[str]:
        """Returns a list of strings with the names of the output columns."""
        return self.output_columns
