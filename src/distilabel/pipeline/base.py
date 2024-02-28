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

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from typing_extensions import Self

from distilabel.pipeline._dag import DAG
from distilabel.pipeline.serialization import _Serializable

if TYPE_CHECKING:
    from os import PathLike

    from distilabel.pipeline.step.base import Step


BASE_CACHE_DIR = Path.home() / ".cache" / "distilabel" / "pipelines"


class _GlobalPipelineManager:
    """Class to manage the global pipeline instance that will be used by the steps when
    created within a pipeline context.

    Attributes:
        _context_global_pipeline: The global pipeline instance.
    """

    _context_global_pipeline: Union["BasePipeline", None] = None

    @classmethod
    def set_pipeline(cls, pipeline: Union["BasePipeline", None] = None) -> None:
        """Set the global pipeline instance.

        Args:
            pipeline: The global pipeline instance.
        """
        cls._context_global_pipeline = pipeline

    @classmethod
    def get_pipeline(cls) -> Union["BasePipeline", None]:
        """Get the global pipeline instance.

        Returns:
            The global pipeline instance.
        """
        return cls._context_global_pipeline


class BasePipeline(_Serializable):
    """Base class for a `distilabel` pipeline.

    Attributes:
        dag: The `DAG` instance that represents the pipeline.
    """

    def __init__(self, cache_dir: Optional["PathLike"] = None) -> None:
        self.dag = DAG()
        self._cache_dir = Path(cache_dir) if cache_dir else BASE_CACHE_DIR

    def __enter__(self) -> Self:
        """Set the global pipeline instance when entering a pipeline context."""
        _GlobalPipelineManager.set_pipeline(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Unset the global pipeline instance when exiting a pipeline context."""
        _GlobalPipelineManager.set_pipeline(None)

    def _create_signature(self) -> str:
        """Makes a signature (hash) of a pipeline, using the step ids and the adjacency between them.

        The main use is to find the pipeline in the cache folder.

        Returns:
            int: Signature of the pipeline.
        """
        hasher = hashlib.sha1()
        # Use the names of the steps plus adjacency to get a hash
        dag_dump = self.dump()["dag"]
        step_ids = [node["id"] for node in dag_dump["nodes"]]
        adjacency_ids = [adj["id"] for adj in itertools.chain(*dag_dump["adjacency"])]
        hasher.update("-".join(step_ids + adjacency_ids).encode())
        return hasher.hexdigest()

    def run(self, parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Run the pipeline. It will set the runtime parameters for the steps and validate
        the pipeline.

        This method should be extended by the specific pipeline implementation,
        adding the logic to run the pipeline.

        Args:
            parameters: A dictionary with the step name as the key and a dictionary with
                the parameter name as the key and the parameter value as the value.
        """
        self._set_runtime_parameters(parameters or {})
        self.dag.validate()

    def _add_step(self, step: "Step") -> None:
        """Add a step to the pipeline.

        Args:
            step: The step to be added to the pipeline.
        """
        self.dag.add_step(step)

    def _add_edge(self, from_step: str, to_step: str) -> None:
        """Add an edge between two steps in the pipeline.

        Args:
            from_step: The name of the step that will generate the input for `to_step`.
            to_step: The name of the step that will receive the input from `from_step`.
        """
        self.dag.add_edge(from_step, to_step)

    def _set_runtime_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> None:
        """Set the runtime parameters for the steps in the pipeline.

        Args:
            parameters: A dictionary with the step name as the key and a dictionary with
            the parameter name as the key and the parameter value as the value.
        """
        for step_name, step_parameters in parameters.items():
            step = self.dag.get_step(step_name)["step"]
            step._set_runtime_parameters(step_parameters)

    def _get_state(self) -> Dict[str, Any]:
        """Obtains the info state of the pipeline.

        MUST BE IMPLEMENTED BY SUBCLASSES.
        """
        return {}

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        from distilabel import __version__

        return {
            "dag": self.dag.dump(),
            "_state_": self._get_state(),
            "_meta_": {"__version__": __version__},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasePipeline":
        """Create a Pipeline from a dict containing the serialized data.

        Note:
            It's intended for internal use.

        Args:
            data (Dict[str, Any]): Dict containing the serialized data from a Pipeline.
        Raises:
            ValueError: _description_

        Returns:
            pipeline (BasePipeline): _description_
        """
        pipe = cls()
        if dag := data.get("dag"):
            # For DAG.from_dict to work we need to be in the context of a Pipeline
            pipe.dag = DAG.from_dict(dag)
            return pipe
        else:
            raise ValueError("No DAG found in the data dictionary")


@dataclass
class _Batch:
    """Dataclass to represent a batch of data to be processed by a `Step`.

    Attributes:
        seq_no: The sequence number of the batch.
        step_name: The name of the step that will process the batch.
        last_batch: A flag to indicate if the batch is the last one.
        data: The data to be processed.
    """

    seq_no: int
    step_name: str
    last_batch: bool
    data: List[List[Dict[str, Any]]] = field(default_factory=list, repr=False)

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

    @classmethod
    def from_batches(cls, step_name: str, batches: List["_Batch"]) -> "_Batch":
        """Create a `_Batch` instance from a list of `_Batch` instances.

        Args:
            step_name: The name of the step that will process the batch.
            batches: A list of `_Batch` instances.

        Returns:
            A `_Batch` instance.
        """

        seq_no = batches[0].seq_no
        if not all(batch.seq_no == seq_no for batch in batches):
            raise ValueError("All batches must have the same sequence number")

        data = [batch.data[0] for batch in batches]
        last_batch = batches[-1].last_batch
        return cls(seq_no, step_name, last_batch, data)


class _BatchManager:
    """Class to manage the batches received from the steps. It keeps track of the
    received batches and returns the list of batches to be processed when all the inputs
    for a step are received.

    Attributes:
        _batches: A dictionary with the step name as the key and a dictionary with the
            predecessor step name as the key and a list of batches as the value.
    """

    def __init__(self, batches: Dict[str, Dict[str, List["_Batch"]]]) -> None:
        """Initialize the `_BatchManager` instance.

        Args:
            batches: A dictionary with the step name as the key and a dictionary with the
                predecessor step name as the key and a list of batches as the value.
        """
        self._batches = batches

    def add_batch(self, to_step: str, batch: _Batch) -> Union[List[_Batch], None]:
        """Add an output batch from to `to_step`. If all the inputs for `to_step` are
        received, then return the list of batches to be processed.

        Args:
            to_step: The name of the step that will process the batch.
            batch: The output batch of an step to be processed by `to_step`.

        Returns:
            If all the inputs for `to_step` are received, then return the list of batches
            to be processed. Otherwise, return `None`.

        Raises:
            ValueError: If a batch from `from_step` to `to_step` with the same sequence
            number was already received.
        """
        from_step = batch.step_name
        for batch in self._batches[to_step][from_step]:
            if batch.seq_no == batch.seq_no:
                raise ValueError(
                    f"A batch from '{from_step}' to '{to_step}' with sequence number "
                    f"{batch.seq_no} was already received"
                )

        self._batches[to_step][from_step].append(batch)

        if self._step_input_batches_received(to_step):
            batches = [batches.pop(0) for batches in self._batches[to_step].values()]
            return batches

        return None

    @classmethod
    def from_dag(cls, dag: "DAG") -> "_BatchManager":
        """Create a `_BatchManager` instance from a `DAG` instance.

        Args:
            dag: The `DAG` instance.

        Returns:
            A `_BatchManager` instance.
        """
        batches = {}
        for step_name in dag:
            # Skip generator steps as they don't have predecessors
            if dag.get_step(step_name)["step"].is_generator:
                continue
            batches[step_name] = {}
            for predecessor in dag.get_step_predecessors(step_name):
                batches[step_name][predecessor] = []
        return cls(batches)

    def _step_input_batches_received(self, step_name: str) -> bool:
        """Check if all the input batches for a step have been received.

        Args:
            step_name: The name of the step.

        Returns:
            A boolean indicating if all the inputs for the step have been received.
        """

        return all(len(batches) > 0 for batches in self._batches[step_name].values())
