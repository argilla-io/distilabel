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

import pytest

from distilabel.pipeline._dag import DAG
from distilabel.steps.generators.data import LoadDataFromDicts


class _DummyStep:
    """Minimal step-like object for DAG testing."""

    def __init__(self, name: str):
        self.name = name


class TestDAGRemoveStep:
    def test_remove_existing_step(self):
        """Step and edges are removed."""
        dag = DAG()

        loader = LoadDataFromDicts(
            name="loader",
            data=[{"x": 1}],
            batch_size=1,
        )
        dag.add_step(loader)

        loader2 = LoadDataFromDicts(
            name="loader2",
            data=[{"x": 2}],
            batch_size=1,
        )
        dag.add_step(loader2)

        assert "loader" in dag.G
        assert "loader2" in dag.G

        dag.remove_step("loader")

        assert "loader" not in dag.G
        assert "loader2" in dag.G
        assert len(dag) == 1

    def test_remove_nonexistent_step_raises(self):
        """ValueError when step doesn't exist."""
        dag = DAG()

        with pytest.raises(ValueError, match="does not exist"):
            dag.remove_step("nonexistent")

    def test_cached_properties_invalidated(self):
        """root_steps/leaf_steps update after removal."""
        dag = DAG()

        loader1 = LoadDataFromDicts(
            name="loader1",
            data=[{"x": 1}],
            batch_size=1,
        )
        loader2 = LoadDataFromDicts(
            name="loader2",
            data=[{"x": 2}],
            batch_size=1,
        )
        dag.add_step(loader1)
        dag.add_step(loader2)

        # Access cached properties to populate them
        assert "loader1" in dag.root_steps
        assert "loader2" in dag.root_steps

        # Remove one step
        dag.remove_step("loader1")

        # Cached properties should be invalidated and recomputed
        assert "loader1" not in dag.root_steps
        assert "loader2" in dag.root_steps
        assert len(dag.root_steps) == 1

    def test_step_names_in_topological_order(self):
        """Verify topological order returns correct order."""
        dag = DAG()

        loader = LoadDataFromDicts(
            name="loader",
            data=[{"x": 1}],
            batch_size=1,
        )
        dag.add_step(loader)

        order = dag.step_names_in_topological_order()
        assert order == ["loader"]
