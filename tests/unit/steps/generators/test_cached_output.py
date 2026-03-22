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

import json
from pathlib import Path

import pytest

from distilabel.steps.generators.cached_output import LoadFromCachedOutput


def _write_batch_json(
    dir_path: Path, seq_no: int, data: list, last_batch: bool = False
) -> None:
    """Write a batch JSON file matching the _Batch serialization format.

    Includes the type_info field required by _Serializable.from_json().
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    batch = {
        "seq_no": seq_no,
        "step_name": "test_step",
        "last_batch": last_batch,
        "data": [data],
        "data_hash": None,
        "accumulated": False,
        "created_from": {},
        "batch_routed_to": [],
        "size": len(data),
        "type_info": {
            "module": "distilabel.pipeline.batch",
            "name": "_Batch",
        },
    }
    (dir_path / f"batch_{seq_no}.json").write_text(json.dumps(batch))


class TestLoadFromCachedOutput:
    def test_outputs_returns_columns(self):
        """Pre-configured columns are returned."""
        step = LoadFromCachedOutput(
            name="test_loader",
            cache_dir="/fake/path",
            output_columns=["col_a", "col_b"],
        )
        assert step.outputs == ["col_a", "col_b"]

    def test_process_yields_records(self, tmp_path):
        """Write batch JSON via _Batch dump format, verify yield format."""
        cache_dir = tmp_path / "step_cache"
        _write_batch_json(
            cache_dir,
            0,
            [{"instruction": "hello", "result": "HELLO"}],
            last_batch=False,
        )
        _write_batch_json(
            cache_dir,
            1,
            [{"instruction": "world", "result": "WORLD"}],
            last_batch=True,
        )

        step = LoadFromCachedOutput(
            name="test_loader",
            cache_dir=str(cache_dir),
            output_columns=["instruction", "result"],
        )
        step.load()

        results = list(step.process())

        assert len(results) == 2
        records_0, is_last_0 = results[0]
        records_1, is_last_1 = results[1]

        assert isinstance(records_0, list)
        assert isinstance(records_0[0], dict)
        assert records_0[0]["instruction"] == "hello"
        assert is_last_0 is False
        assert records_1[0]["instruction"] == "world"
        assert is_last_1 is True

    def test_process_with_offset(self, tmp_path):
        """Skip N batch files when offset is provided."""
        cache_dir = tmp_path / "step_cache"
        for i in range(3):
            _write_batch_json(
                cache_dir,
                i,
                [{"val": i}],
                last_batch=(i == 2),
            )

        step = LoadFromCachedOutput(
            name="test_loader",
            cache_dir=str(cache_dir),
            output_columns=["val"],
        )
        step.load()

        results = list(step.process(offset=1))
        assert len(results) == 2  # batches 1 and 2
        assert results[0][0][0]["val"] == 1
        assert results[1][0][0]["val"] == 2

    def test_missing_dir_raises(self, tmp_path):
        """ValueError on nonexistent path."""
        step = LoadFromCachedOutput(
            name="test_loader",
            cache_dir=str(tmp_path / "nonexistent"),
            output_columns=["col"],
        )
        with pytest.raises(ValueError, match="does not exist"):
            step.load()

    def test_no_batch_files_raises(self, tmp_path):
        """ValueError on empty dir (no batch files)."""
        empty_dir = tmp_path / "empty_cache"
        empty_dir.mkdir()

        step = LoadFromCachedOutput(
            name="test_loader",
            cache_dir=str(empty_dir),
            output_columns=["col"],
        )
        with pytest.raises(ValueError, match="No batch files"):
            step.load()

    def test_auto_detect_columns(self, tmp_path):
        """When output_columns is empty, auto-detect from first batch."""
        cache_dir = tmp_path / "step_cache"
        _write_batch_json(
            cache_dir,
            0,
            [{"auto_col_a": 1, "auto_col_b": 2}],
            last_batch=True,
        )

        step = LoadFromCachedOutput(
            name="test_loader",
            cache_dir=str(cache_dir),
        )
        step.load()

        assert "auto_col_a" in step.outputs
        assert "auto_col_b" in step.outputs
