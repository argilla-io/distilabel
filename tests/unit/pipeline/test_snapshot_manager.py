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

import os
import time
from pathlib import Path

import pytest

from distilabel.pipeline.snapshot_manager import CacheSnapshotManager


class TestCacheSnapshotManager:
    def test_snapshot_start_stop(self, tmp_path):
        """Lifecycle: start and stop without errors."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "test.json").write_text("{}")

        snapshots_dir = tmp_path / "snapshots"

        manager = CacheSnapshotManager(
            cache_dir=cache_dir,
            snapshots_dir=snapshots_dir,
            pipeline_name="test_pipe",
            snapshot_interval_seconds=3600,
        )
        manager.start()
        manager.stop()

        assert snapshots_dir.exists()

    def test_snapshot_creation(self, tmp_path):
        """With 1s interval, verify snapshot dir created with files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "data.json").write_text('{"key": "value"}')

        snapshots_dir = tmp_path / "snapshots"

        manager = CacheSnapshotManager(
            cache_dir=cache_dir,
            snapshots_dir=snapshots_dir,
            pipeline_name="test_pipe",
            snapshot_interval_seconds=1,
        )
        manager.start()
        time.sleep(2.5)  # Wait for at least one snapshot cycle
        manager.stop()

        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        assert len(snapshot_dirs) >= 1
        # Check that the snapshot contains the file
        first_snapshot = snapshot_dirs[0]
        assert (first_snapshot / "data.json").exists()

    def test_final_snapshot(self, tmp_path):
        """Verify take_final_snapshot() creates a snapshot."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "file.txt").write_text("content")

        snapshots_dir = tmp_path / "snapshots"

        manager = CacheSnapshotManager(
            cache_dir=cache_dir,
            snapshots_dir=snapshots_dir,
            pipeline_name="test_pipe",
            snapshot_interval_seconds=3600,
        )
        manager.start()

        result = manager.take_final_snapshot()
        manager.stop()

        assert result is True
        assert manager.snapshot_count >= 1
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        assert len(snapshot_dirs) >= 1

    def test_empty_cache_skipped(self, tmp_path):
        """No source files -> no snapshot created."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()  # Empty directory

        snapshots_dir = tmp_path / "snapshots"

        manager = CacheSnapshotManager(
            cache_dir=cache_dir,
            snapshots_dir=snapshots_dir,
            pipeline_name="test_pipe",
            snapshot_interval_seconds=3600,
        )
        manager.start()

        result = manager.take_final_snapshot()
        manager.stop()

        assert result is False
        assert manager.snapshot_count == 0

    def test_cleanup_old_snapshots_removes_expired(self, tmp_path):
        """Create 3 snapshots with faked mtimes, set retention_days=1,
        verify only the 2 oldest are deleted."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "data.txt").write_text("x")

        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        # Create 3 fake snapshots
        now = time.time()
        for i, name in enumerate(["old1", "old2", "recent"]):
            snap = snapshots_dir / name
            snap.mkdir()
            (snap / "file.txt").write_text("data")
            if name.startswith("old"):
                # Backdate to 5 days ago
                old_time = now - (5 * 86400)
                os.utime(str(snap), (old_time, old_time))
            # "recent" keeps current mtime

        manager = CacheSnapshotManager(
            cache_dir=cache_dir,
            snapshots_dir=snapshots_dir,
            pipeline_name="test_pipe",
            snapshot_interval_seconds=3600,
            retention_days=1,
        )

        deleted = manager._cleanup_old_snapshots()

        assert len(deleted) == 2
        assert "old1" in deleted or "old2" in deleted
        # "recent" should still exist
        assert (snapshots_dir / "recent").exists()

    def test_cleanup_always_keeps_most_recent(self, tmp_path):
        """Single old snapshot should NOT be deleted (always keep at least one)."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        snap = snapshots_dir / "only_one"
        snap.mkdir()
        (snap / "file.txt").write_text("data")
        old_time = time.time() - (60 * 86400)
        os.utime(str(snap), (old_time, old_time))

        manager = CacheSnapshotManager(
            cache_dir=tmp_path / "cache",
            snapshots_dir=snapshots_dir,
            pipeline_name="test_pipe",
            snapshot_interval_seconds=3600,
            retention_days=1,
        )

        deleted = manager._cleanup_old_snapshots()

        assert len(deleted) == 0
        assert (snapshots_dir / "only_one").exists()

    def test_cleanup_skipped_when_no_retention(self, tmp_path):
        """retention_days=None -> no deletions."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        snap = snapshots_dir / "some_snap"
        snap.mkdir()
        (snap / "file.txt").write_text("data")
        old_time = time.time() - (60 * 86400)
        os.utime(str(snap), (old_time, old_time))

        manager = CacheSnapshotManager(
            cache_dir=tmp_path / "cache",
            snapshots_dir=snapshots_dir,
            pipeline_name="test_pipe",
            snapshot_interval_seconds=3600,
            retention_days=None,
        )

        deleted = manager._cleanup_old_snapshots()

        assert len(deleted) == 0
        assert (snapshots_dir / "some_snap").exists()

    def test_cleanup_preserves_recent_snapshots(self, tmp_path):
        """All snapshots within retention window -> none deleted."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        for name in ["snap1", "snap2", "snap3"]:
            snap = snapshots_dir / name
            snap.mkdir()
            (snap / "file.txt").write_text("data")
            # All have current mtime (within any retention window)

        manager = CacheSnapshotManager(
            cache_dir=tmp_path / "cache",
            snapshots_dir=snapshots_dir,
            pipeline_name="test_pipe",
            snapshot_interval_seconds=3600,
            retention_days=30,
        )

        deleted = manager._cleanup_old_snapshots()

        assert len(deleted) == 0
        assert len(list(snapshots_dir.iterdir())) == 3
