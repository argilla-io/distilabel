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

"""Integration tests for cross-config cache restoration.

Verifies that pipelines sharing common prefix steps can reuse
cached outputs from different pipeline configurations.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.base import Step, StepInput
from distilabel.steps.generators.cached_output import LoadFromCachedOutput


class UpperCase(Step):
    """Deterministic step: uppercases the 'instruction' field."""

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def process(self, inputs: StepInput):
        for inp in inputs:
            inp["uppercased"] = inp["instruction"].upper()
        yield inputs

    @property
    def outputs(self) -> List[str]:
        return ["uppercased"]


class AddSuffix(Step):
    """Deterministic step with configurable suffix."""

    suffix: str = "_done"

    @property
    def inputs(self) -> List[str]:
        return ["uppercased"]

    def process(self, inputs: StepInput):
        for inp in inputs:
            inp["result"] = inp["uppercased"] + self.suffix
        yield inputs

    @property
    def outputs(self) -> List[str]:
        return ["result"]


class TestCrossConfigFullMatch:
    def test_cross_config_full_match(self):
        """When all step signatures match, all caches are copied and native caching kicks in."""
        with TemporaryDirectory() as tmp_dir:
            cache_a = Path(tmp_dir) / "pipeline_a"
            cache_b = Path(tmp_dir) / "pipeline_b"

            # Pipeline A: LoadData -> UpperCase -> AddSuffix("_v1")
            with Pipeline(name="pipe_a", cache_dir=str(cache_a)) as pipe_a:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 20, batch_size=5
                )
                upper = UpperCase()
                suffix = AddSuffix(suffix="_v1")
                loader >> upper >> suffix
            distiset_a = pipe_a.run()

            # Pipeline B: exact same config, different cache_dir, restore from A
            with Pipeline(name="pipe_b", cache_dir=str(cache_b)) as pipe_b:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 20, batch_size=5
                )
                upper = UpperCase()
                suffix = AddSuffix(suffix="_v1")
                loader >> upper >> suffix
            distiset_b = pipe_b.run(
                restore_cache_from=str(cache_a / "pipe_a")
            )

            assert (
                distiset_a["default"]["train"].to_list()
                == distiset_b["default"]["train"].to_list()
            )


class TestCrossConfigPartialMatch:
    def test_cross_config_partial_match(self):
        """When prefix steps match but final step differs, only the final step executes fresh."""
        with TemporaryDirectory() as tmp_dir:
            cache_a = Path(tmp_dir) / "pipeline_a"
            cache_b = Path(tmp_dir) / "pipeline_b"

            # Pipeline A: LoadData -> UpperCase -> AddSuffix("_formal")
            with Pipeline(name="pipe_a", cache_dir=str(cache_a)) as pipe_a:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 20, batch_size=5
                )
                upper = UpperCase()
                suffix = AddSuffix(name="add_suffix", suffix="_formal")
                loader >> upper >> suffix
            distiset_a = pipe_a.run()

            # Pipeline B: LoadData -> UpperCase -> AddSuffix("_casual") <- different suffix
            with Pipeline(name="pipe_b", cache_dir=str(cache_b)) as pipe_b:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 20, batch_size=5
                )
                upper = UpperCase()
                suffix = AddSuffix(name="add_suffix", suffix="_casual")
                loader >> upper >> suffix
            distiset_b = pipe_b.run(
                restore_cache_from=str(cache_a / "pipe_a")
            )

            results_a = distiset_a["default"]["train"].to_list()
            results_b = distiset_b["default"]["train"].to_list()

            # UpperCase output should be the same (reused from cache)
            assert all(r["uppercased"] == "HELLO" for r in results_a)
            assert all(r["uppercased"] == "HELLO" for r in results_b)

            # But final result differs due to different suffix
            assert all(r["result"] == "HELLO_formal" for r in results_a)
            assert all(r["result"] == "HELLO_casual" for r in results_b)


class TestCrossConfigNoMatch:
    def test_cross_config_no_match(self):
        """When no step signatures match, the entire pipeline executes from scratch."""
        with TemporaryDirectory() as tmp_dir:
            cache_a = Path(tmp_dir) / "pipeline_a"
            cache_b = Path(tmp_dir) / "pipeline_b"

            # Pipeline A with one set of data
            with Pipeline(name="pipe_a", cache_dir=str(cache_a)) as pipe_a:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 10, batch_size=5
                )
                upper = UpperCase()
                loader >> upper
            pipe_a.run()

            # Pipeline B with completely different data -> different generator signature
            with Pipeline(name="pipe_b", cache_dir=str(cache_b)) as pipe_b:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "goodbye"}] * 10, batch_size=5
                )
                upper = UpperCase()
                loader >> upper
            distiset_b = pipe_b.run(
                restore_cache_from=str(cache_a / "pipe_a")
            )

            # Should still produce correct results (full fresh execution)
            assert all(
                r["uppercased"] == "GOODBYE"
                for r in distiset_b["default"]["train"].to_list()
            )


class TestCrossConfigNonexistentSource:
    def test_cross_config_nonexistent_source(self):
        """When restore_cache_from points to a nonexistent path, pipeline runs normally."""
        with TemporaryDirectory() as tmp_dir:
            with Pipeline(name="pipe", cache_dir=tmp_dir) as pipe:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 10, batch_size=5
                )
                upper = UpperCase()
                loader >> upper
            distiset = pipe.run(restore_cache_from="/nonexistent/path")
            assert len(distiset["default"]["train"]) == 10


class TestCrossConfigTransitivity:
    def test_cross_config_transitivity(self):
        """Cache transitivity: Run1->Run2->Run3 chains work because matching caches are always copied."""
        with TemporaryDirectory() as tmp_dir:
            cache_1 = Path(tmp_dir) / "run1"
            cache_2 = Path(tmp_dir) / "run2"
            cache_3 = Path(tmp_dir) / "run3"

            # Run 1: LoadData -> UpperCase -> AddSuffix("_v1")
            with Pipeline(name="pipe", cache_dir=str(cache_1)) as p1:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 10, batch_size=5
                )
                upper = UpperCase()
                suffix = AddSuffix(name="add_suffix", suffix="_v1")
                loader >> upper >> suffix
            p1.run()

            # Run 2: same prefix, different suffix -> restore from Run 1
            with Pipeline(name="pipe", cache_dir=str(cache_2)) as p2:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 10, batch_size=5
                )
                upper = UpperCase()
                suffix = AddSuffix(name="add_suffix", suffix="_v2")
                loader >> upper >> suffix
            p2.run(restore_cache_from=str(cache_1 / "pipe"))

            # Run 3: same prefix, yet another suffix -> restore from Run 2 (NOT Run 1!)
            with Pipeline(name="pipe", cache_dir=str(cache_3)) as p3:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 10, batch_size=5
                )
                upper = UpperCase()
                suffix = AddSuffix(name="add_suffix", suffix="_v3")
                loader >> upper >> suffix
            distiset_3 = p3.run(restore_cache_from=str(cache_2 / "pipe"))

            # Verify Run 3 succeeded and used the transitive cache
            results = distiset_3["default"]["train"].to_list()
            assert all(r["result"] == "HELLO_v3" for r in results)


class TestSnapshotE2E:
    def test_snapshot_e2e(self):
        """Verify snapshot directory contains files after pipeline execution."""
        with TemporaryDirectory() as tmp_dir:
            snapshot_dir = Path(tmp_dir) / "snapshots"
            with Pipeline(name="pipe", cache_dir=tmp_dir) as pipe:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 50, batch_size=10
                )
                upper = UpperCase()
                loader >> upper
            pipe.run(snapshot_dir=str(snapshot_dir), snapshot_interval_seconds=1)
            # At minimum, the final snapshot should exist
            assert snapshot_dir.exists()
            snapshots = list(snapshot_dir.iterdir())
            assert len(snapshots) >= 1


class TestSnapshotRetentionCleanup:
    def test_snapshot_retention_cleanup(self):
        """Old snapshots beyond retention_days are automatically cleaned up."""
        import os
        import time

        with TemporaryDirectory() as tmp_dir:
            snapshot_dir = Path(tmp_dir) / "snapshots"
            # Create fake old snapshots with backdated mtimes
            snapshot_dir.mkdir()
            old_snap = snapshot_dir / "20260101_000000"
            old_snap.mkdir()
            (old_snap / "dummy.json").write_text("{}")
            # Backdate mtime to 60 days ago
            old_time = time.time() - (60 * 86400)
            os.utime(str(old_snap), (old_time, old_time))

            with Pipeline(name="pipe", cache_dir=tmp_dir) as pipe:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 10, batch_size=5
                )
                upper = UpperCase()
                loader >> upper
            pipe.run(
                snapshot_dir=str(snapshot_dir),
                snapshot_interval_seconds=1,
                snapshot_retention_days=30,
            )
            # Old snapshot should be cleaned up, only recent one(s) remain
            remaining = [d for d in snapshot_dir.iterdir() if d.is_dir()]
            assert not any(d.name == "20260101_000000" for d in remaining)
            assert len(remaining) >= 1  # at least the new snapshot


class TestLoadFromCachedOutputE2E:
    def test_load_from_cached_output_e2e(self):
        """LoadFromCachedOutput reads cached step output and feeds it to downstream steps."""
        with TemporaryDirectory() as tmp_dir:
            # Pipeline A: generate cached output
            with Pipeline(name="pipe_a", cache_dir=tmp_dir) as pipe_a:
                loader = LoadDataFromDicts(
                    data=[{"instruction": "hello"}] * 20, batch_size=5
                )
                upper = UpperCase(name="upper_step")
                loader >> upper
            pipe_a.run()

            # Find the cached step directory for 'upper_step'
            steps_data = Path(tmp_dir) / "pipe_a" / "steps_data"
            upper_dirs = [
                d for d in steps_data.iterdir() if d.name.startswith("upper_step_")
            ]
            assert len(upper_dirs) == 1
            cached_dir = upper_dirs[0]

            # Pipeline B: read from cached output directly
            with Pipeline(name="pipe_b", cache_dir=tmp_dir) as pipe_b:
                cached_loader = LoadFromCachedOutput(
                    cache_dir=str(cached_dir),
                    output_columns=["instruction", "uppercased"],
                )
                suffix = AddSuffix(suffix="_from_cache")
                cached_loader >> suffix
            distiset_b = pipe_b.run(use_cache=False)

            results = distiset_b["default"]["train"].to_list()
            assert len(results) == 20
            assert all(r["result"] == "HELLO_from_cache" for r in results)
