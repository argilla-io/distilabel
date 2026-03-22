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
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class CacheSnapshotManager:
    """Background daemon thread that periodically copies cache dir to snapshots.

    Includes retention-based cleanup of old snapshots.

    Attributes:
        _cache_dir: The cache directory to snapshot.
        _snapshots_dir: The directory where snapshots will be stored.
        _pipeline_name: The name of the pipeline.
        _interval: The interval in seconds between snapshots.
        _retention_days: Number of days to retain snapshots. None means no cleanup.
        _stop_event: Threading event to signal the snapshot loop to stop.
        _thread: The background daemon thread.
        _snapshot_count: Number of snapshots created.
    """

    def __init__(
        self,
        cache_dir: Path,
        snapshots_dir: Path,
        pipeline_name: str,
        snapshot_interval_seconds: int = 57600,
        retention_days: Optional[int] = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._snapshots_dir = snapshots_dir
        self._pipeline_name = pipeline_name
        self._interval = snapshot_interval_seconds
        self._retention_days = retention_days
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshot_count = 0

    def start(self) -> None:
        """Launch daemon thread running _snapshot_loop."""
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._snapshot_loop, daemon=True)
        self._thread.start()
        self._safe_log(logging.INFO, "Snapshot manager started")

    def stop(self) -> None:
        """Signal stop, join thread with 30s timeout."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)
        self._safe_log(
            logging.INFO,
            f"Snapshot manager stopped. Total snapshots created: {self._snapshot_count}",
        )

    def take_final_snapshot(self) -> bool:
        """Create one final snapshot before pipeline exits.

        Returns:
            True if the snapshot was created successfully, False otherwise.
        """
        snapshot_path = self._create_snapshot()
        return snapshot_path is not None

    @property
    def snapshot_count(self) -> int:
        """Return the total number of snapshots created."""
        return self._snapshot_count

    def _snapshot_loop(self) -> None:
        """Wait interval, create snapshot, cleanup old ones, repeat until stopped."""
        while not self._stop_event.wait(timeout=self._interval):
            self._create_snapshot()
            self._cleanup_old_snapshots()

    def _create_snapshot(self) -> Optional[Path]:
        """Copy cache_dir to snapshots_dir/{timestamp}/ using shutil.copytree.

        Returns:
            The path to the created snapshot, or None if creation failed.
        """
        if not self._cache_dir.exists():
            self._safe_log(
                logging.DEBUG,
                f"Cache directory does not exist: {self._cache_dir}, skipping snapshot",
            )
            return None

        # Check if there's anything to snapshot
        has_files = any(self._cache_dir.rglob("*"))
        if not has_files:
            self._safe_log(
                logging.DEBUG,
                "Cache directory is empty, skipping snapshot",
            )
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self._snapshots_dir / timestamp

        try:
            shutil.copytree(str(self._cache_dir), str(snapshot_path))
            self._snapshot_count += 1
            self._safe_log(
                logging.INFO,
                f"Snapshot #{self._snapshot_count} created at: {snapshot_path}",
            )
            return snapshot_path
        except Exception as e:
            self._safe_log(
                logging.ERROR,
                f"Failed to create snapshot: {e}",
            )
            return None

    def _cleanup_old_snapshots(self) -> List[str]:
        """Remove snapshot directories older than retention_days.

        1. List all snapshot dirs in snapshots_dir, sorted by mtime descending
        2. Always keep at least one snapshot (the most recent), regardless of age
        3. Delete snapshots with mtime older than (now - retention_days)
        4. Log all deletions for auditability
        5. Return list of deleted directory names

        Only runs if retention_days is set (not None).

        Returns:
            List of deleted snapshot directory names.
        """
        if self._retention_days is None:
            return []

        deleted: List[str] = []

        snapshot_dirs = [
            d for d in self._snapshots_dir.iterdir() if d.is_dir()
        ]
        if len(snapshot_dirs) <= 1:
            return deleted

        # Sort by mtime descending (most recent first)
        snapshot_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

        cutoff_time = time.time() - (self._retention_days * 86400)

        # Skip the first one (most recent) — always keep it
        for snapshot_dir in snapshot_dirs[1:]:
            if snapshot_dir.stat().st_mtime < cutoff_time:
                try:
                    shutil.rmtree(str(snapshot_dir))
                    deleted.append(snapshot_dir.name)
                    self._safe_log(
                        logging.INFO,
                        f"Cleaned up old snapshot: {snapshot_dir.name}",
                    )
                except Exception as e:
                    self._safe_log(
                        logging.ERROR,
                        f"Failed to clean up snapshot {snapshot_dir.name}: {e}",
                    )

        return deleted

    def _safe_log(self, level: int, msg: str) -> None:
        """Log safely — catches errors when QueueHandler is already closed."""
        try:
            logger = logging.getLogger("distilabel.pipeline.snapshot")
            logger.log(level, msg)
        except Exception:
            pass
