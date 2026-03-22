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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from distilabel.utils.cache import (
    copy_directory_with_stats,
    find_cache_root,
    find_latest_snapshot,
)

if TYPE_CHECKING:
    from distilabel.pipeline._dag import DAG

logger = logging.getLogger("distilabel.pipeline.cache_restore")


@dataclass
class CacheRestoreResult:
    """Result of a cross-config cache restoration operation.

    Attributes:
        success: Whether the restoration was successful.
        error: Error message if the restoration failed.
        steps_restored: Number of steps whose caches were restored.
        steps_skipped: Number of steps that were skipped (no matching cache).
        first_missing_step: The name of the first step whose cache was not found.
        resumption_step: The first step that didn't match (where fresh execution starts).
        last_cached_step_dir: Path to the TARGET directory of the last matching step's
            cache (after copy), used by LoadFromCachedOutput.
        output_columns: Column names from the last cached step's output.
    """

    success: bool
    error: Optional[str] = None
    steps_restored: int = 0
    steps_skipped: int = 0
    first_missing_step: Optional[str] = None
    resumption_step: Optional[str] = None
    last_cached_step_dir: Optional[str] = None
    output_columns: List[str] = field(default_factory=list)


def compute_step_signatures(dag: "DAG") -> Dict[str, str]:
    """Compute {step_name: step.signature} for all steps in topological order.

    Uses dag.step_names_in_topological_order() + step.signature from SignatureMixin.
    Result dict preserves topo order (Python 3.7+ dict insertion order).

    Args:
        dag: The pipeline DAG.

    Returns:
        A dictionary mapping step names to their signatures, in topological order.
    """
    from distilabel.constants import STEP_ATTR_NAME

    step_signatures = {}
    for step_name in dag.step_names_in_topological_order():
        step = dag.get_step(step_name)[STEP_ATTR_NAME]
        step_signatures[step_name] = step.signature
    return step_signatures


def _restore_cache_sequential(
    source_steps_data: Path,
    target_steps_data: Path,
    step_signatures: Dict[str, str],
) -> CacheRestoreResult:
    """Sequential signature matching for cache restoration.

    For each step in topological order:
    1. Look for {step_name}_{signature}/ in source_steps_data
    2. If found: copy batch files to target, increment steps_restored
    3. If not found: record as first_missing_step, set resumption_step, stop
    4. If ALL match: all caches copied, native caching will handle the rest

    Cache transitivity: ALL matching step caches are copied to the target directory.
    Without this, when pipelines chain through intermediate runs (Run1->Run2->Run3),
    Run2's snapshots won't contain Run1's cached steps, breaking transitivity.

    Args:
        source_steps_data: Path to the source steps_data directory.
        target_steps_data: Path to the target steps_data directory.
        step_signatures: Dict mapping step names to signatures in topological order.

    Returns:
        A CacheRestoreResult with the restoration outcome.
    """
    result = CacheRestoreResult(success=True)
    last_cached_dir = None
    output_columns: List[str] = []

    step_names = list(step_signatures.keys())

    for i, step_name in enumerate(step_names):
        signature = step_signatures[step_name]
        step_dir_name = f"{step_name}_{signature}"
        source_dir = source_steps_data / step_dir_name
        target_dir = target_steps_data / step_dir_name

        if source_dir.exists() and source_dir.is_dir():
            # Copy batch files to target (cache transitivity)
            if not target_dir.exists():
                files_copied, _ = copy_directory_with_stats(source_dir, target_dir)
                logger.info(
                    f"Restored cache for step '{step_name}': {files_copied} files copied"
                )
            else:
                logger.info(
                    f"Cache for step '{step_name}' already exists in target, skipping copy"
                )
            result.steps_restored += 1
            last_cached_dir = str(target_dir)

            # Read output columns from the last cached step
            batch_files = sorted(source_dir.glob("batch_*.json"))
            if batch_files:
                from distilabel.pipeline.batch import _Batch

                first_batch = _Batch.from_json(batch_files[0])
                if first_batch.data and first_batch.data[0]:
                    output_columns = list(first_batch.data[0][0].keys())
        else:
            result.first_missing_step = step_name
            result.resumption_step = step_name
            result.steps_skipped = len(step_names) - i
            break

    result.last_cached_step_dir = last_cached_dir
    result.output_columns = output_columns

    # If no steps were restored at all, it's still "successful" but nothing to resume from
    if result.steps_restored == 0:
        result.success = True
        result.resumption_step = None
        result.last_cached_step_dir = None

    return result


def restore_cache_from_input(
    input_cache_path: Path,
    target_cache_dir: Path,
    pipeline_name: str,
    step_signatures: Dict[str, str],
) -> CacheRestoreResult:
    """Entry point for cross-config cache restoration.

    Resolves source path (handles snapshots via find_latest_snapshot),
    finds steps_data dir (via find_cache_root), delegates to _restore_cache_sequential.

    Args:
        input_cache_path: Path to the source cache (may be a pipeline cache dir or snapshot).
        target_cache_dir: The base cache directory for the target pipeline.
        pipeline_name: Name of the target pipeline.
        step_signatures: Dict mapping step names to signatures in topological order.

    Returns:
        A CacheRestoreResult with the restoration outcome.
    """
    if not input_cache_path.exists():
        logger.warning(
            f"Source cache path does not exist: {input_cache_path}. "
            "Pipeline will run from scratch."
        )
        return CacheRestoreResult(
            success=False,
            error=f"Source cache path does not exist: {input_cache_path}",
        )

    # Try to find steps_data directly
    cache_root = find_cache_root(input_cache_path, pipeline_name)

    # If not found directly, check for snapshots
    if cache_root is None:
        snapshot = find_latest_snapshot(input_cache_path)
        if snapshot:
            cache_root = find_cache_root(snapshot, pipeline_name)

    if cache_root is None:
        logger.warning(
            f"Could not find steps_data in source cache: {input_cache_path}. "
            "Pipeline will run from scratch."
        )
        return CacheRestoreResult(
            success=False,
            error=f"Could not find steps_data in source cache: {input_cache_path}",
        )

    source_steps_data = cache_root / "steps_data"
    target_steps_data = target_cache_dir / pipeline_name / "steps_data"
    target_steps_data.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Restoring cache from '{source_steps_data}' to '{target_steps_data}'"
    )

    return _restore_cache_sequential(
        source_steps_data=source_steps_data,
        target_steps_data=target_steps_data,
        step_signatures=step_signatures,
    )
