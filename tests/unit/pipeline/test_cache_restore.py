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

from distilabel.pipeline.cache_restore import (
    CacheRestoreResult,
    _restore_cache_sequential,
    compute_step_signatures,
    restore_cache_from_input,
)


def _write_batch_file(dir_path: Path, seq_no: int, data: list) -> None:
    """Helper to write a fake batch JSON file in the format _Batch expects.

    Includes the type_info field required by _Serializable.from_json().
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    batch_data = {
        "seq_no": seq_no,
        "step_name": "test_step",
        "last_batch": seq_no == 0,
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
    filepath = dir_path / f"batch_{seq_no}.json"
    filepath.write_text(json.dumps(batch_data))


class TestCacheRestoreResult:
    def test_cache_restore_result_defaults(self):
        result = CacheRestoreResult(success=True)
        assert result.success is True
        assert result.error is None
        assert result.steps_restored == 0
        assert result.steps_skipped == 0
        assert result.first_missing_step is None
        assert result.resumption_step is None
        assert result.last_cached_step_dir is None
        assert result.output_columns == []


class TestRestoreCacheSequential:
    def test_restore_cache_sequential_all_match(self, tmp_path):
        """When all step signatures match in source, all caches are copied."""
        source = tmp_path / "source" / "steps_data"
        target = tmp_path / "target" / "steps_data"
        target.mkdir(parents=True)

        # Create source step dirs with matching signatures
        step_sigs = {"step_a": "sig_a", "step_b": "sig_b"}
        for step_name, sig in step_sigs.items():
            _write_batch_file(
                source / f"{step_name}_{sig}",
                0,
                [{"col1": "val1"}],
            )

        result = _restore_cache_sequential(source, target, step_sigs)

        assert result.success is True
        assert result.steps_restored == 2
        assert result.first_missing_step is None
        assert result.resumption_step is None
        assert (target / "step_a_sig_a" / "batch_0.json").exists()
        assert (target / "step_b_sig_b" / "batch_0.json").exists()

    def test_restore_cache_sequential_partial_match(self, tmp_path):
        """When only prefix steps match, resumption metadata is set correctly."""
        source = tmp_path / "source" / "steps_data"
        target = tmp_path / "target" / "steps_data"
        target.mkdir(parents=True)

        step_sigs = {"step_a": "sig_a", "step_b": "sig_b", "step_c": "sig_c"}
        # Only create step_a in source
        _write_batch_file(source / "step_a_sig_a", 0, [{"col1": "val1"}])

        result = _restore_cache_sequential(source, target, step_sigs)

        assert result.success is True
        assert result.steps_restored == 1
        assert result.first_missing_step == "step_b"
        assert result.resumption_step == "step_b"
        assert result.steps_skipped == 2
        assert (target / "step_a_sig_a" / "batch_0.json").exists()

    def test_restore_cache_sequential_no_match(self, tmp_path):
        """When first step doesn't match, nothing is restored."""
        source = tmp_path / "source" / "steps_data"
        target = tmp_path / "target" / "steps_data"
        source.mkdir(parents=True)
        target.mkdir(parents=True)

        step_sigs = {"step_a": "sig_a", "step_b": "sig_b"}

        result = _restore_cache_sequential(source, target, step_sigs)

        assert result.success is True
        assert result.steps_restored == 0
        assert result.first_missing_step == "step_a"
        assert result.resumption_step is None  # No steps restored, so no resumption
        assert result.last_cached_step_dir is None

    def test_restore_cache_sequential_copies_to_target(self, tmp_path):
        """Verify matching step caches are physically copied to target (transitivity)."""
        source = tmp_path / "source" / "steps_data"
        target = tmp_path / "target" / "steps_data"
        target.mkdir(parents=True)

        step_sigs = {"step_a": "sig_a", "step_b": "sig_b"}
        _write_batch_file(source / "step_a_sig_a", 0, [{"x": 1}])
        _write_batch_file(source / "step_b_sig_b", 0, [{"y": 2}])

        result = _restore_cache_sequential(source, target, step_sigs)

        # Both should be physically present in target
        assert (target / "step_a_sig_a" / "batch_0.json").exists()
        assert (target / "step_b_sig_b" / "batch_0.json").exists()
        assert result.steps_restored == 2


class TestRestoreCacheFromInput:
    def test_restore_cache_from_input_with_snapshots(self, tmp_path):
        """When source has a snapshots subdirectory with steps_data."""
        source = tmp_path / "source"
        snapshot = source / "snapshot_20260101"
        steps_data = snapshot / "steps_data"

        step_sigs = {"step_a": "sig_a"}
        _write_batch_file(steps_data / "step_a_sig_a", 0, [{"col": "val"}])

        target = tmp_path / "target"
        result = restore_cache_from_input(source, target, "test_pipe", step_sigs)

        assert result.success is True
        assert result.steps_restored == 1

    def test_restore_cache_from_input_nonexistent(self, tmp_path):
        """When source path doesn't exist, return failure result."""
        result = restore_cache_from_input(
            tmp_path / "nonexistent",
            tmp_path / "target",
            "test_pipe",
            {"step_a": "sig_a"},
        )

        assert result.success is False
        assert "does not exist" in result.error


class TestComputeStepSignatures:
    def test_compute_step_signatures(self):
        """Build a simple DAG and verify signatures are computed."""
        from distilabel.pipeline._dag import DAG
        from distilabel.steps.generators.data import LoadDataFromDicts

        dag = DAG()
        loader = LoadDataFromDicts(
            name="loader",
            data=[{"instruction": "test"}],
            batch_size=5,
        )
        dag.add_step(loader)

        sigs = compute_step_signatures(dag)

        assert "loader" in sigs
        assert isinstance(sigs["loader"], str)
        assert len(sigs["loader"]) > 0
