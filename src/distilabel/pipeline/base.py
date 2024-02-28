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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from typing_extensions import Self

from distilabel.pipeline._dag import DAG
from distilabel.pipeline.logging import get_logger
from distilabel.pipeline.serialization import _Serializable

if TYPE_CHECKING:
    from distilabel.pipeline.step.base import Step


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

    def __init__(self) -> None:
        self.dag = DAG()
        self._logger = get_logger("pipeline")

    def __enter__(self) -> Self:
        """Set the global pipeline instance when entering a pipeline context."""
        _GlobalPipelineManager.set_pipeline(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Unset the global pipeline instance when exiting a pipeline context."""
        _GlobalPipelineManager.set_pipeline(None)

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

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"dag": self.dag.dump()}

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
        """Create a `_Batch` instance with the outputs from the list of batches that
        were received from another steps. All the batches must have the same sequence
        number.

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

    @classmethod
    def accumulate(cls, step_name: str, batches: List[List["_Batch"]]) -> "_Batch":
        """Creates a `_Batch` instance using the data from the list of batches that
        were received from another steps. The batches will be accumulated in a single
        list of data.

        Args:
            step_name: The name of the step that will process the batch.
            batches: a list containing the list of batches received from the predecessors.

        Returns:
            A `_Batch` instance.
        """

        data = []
        for step_batches in batches:
            accumulated_data = [row for batch in step_batches for row in batch.data[0]]
            data.append(accumulated_data)
        return cls(0, step_name, True, data)


class _BatchManager:
    """Class to manage the batches received from the steps. It keeps track of the
    received batches and returns the list of batches to be processed when all the inputs
    for a step are received.

    Attributes:
        _batches: A dictionary with the step name as the key and a dictionary with the
            predecessor step name as the key and a list of batches as the value.
    """

    def __init__(
        self,
        batches: Dict[str, Dict[str, List["_Batch"]]],
        accumulate_batches_for_steps: List[str],
    ) -> None:
        """Initialize the `_BatchManager` instance.

        Args:
            batches: A dictionary with the step name as the key and a dictionary with the
                predecessor step name as the key and a list of batches as the value.
            accumulate_batches_for_steps: A list of steps that will accumulate the
                received batches.
        """
        self._batches = batches
        self._accumulate_batches_for_steps = accumulate_batches_for_steps

    def add_batch(self, to_step: str, batch: _Batch) -> Union[_Batch, None]:
        """Add an output batch from to `to_step`. If all the inputs for `to_step` are
        received, then return the list of batches to be processed.

        Args:
            to_step: The name of the step that will process the batch.
            batch: The output batch of an step to be processed by `to_step`.

        Returns:
            If all the inputs for `to_step` are received, then return a new `_Batch` with
            the data to be processed from the received batches. Otherwise, return `None`.

        Raises:
            ValueError: If a batch from `from_step` to `to_step` with the same sequence
            number was already received.
        """

        from_step = batch.step_name

        batches = self._batches[to_step][from_step]
        if len(batches) > 1:
            last_batch_received = batches[-1]
            if last_batch_received.seq_no > batch.seq_no:
                raise ValueError(
                    f"A batch from '{from_step}' to '{to_step}' with sequence number "
                    f"{last_batch_received.seq_no} was already received"
                )

            expected_seq_no = last_batch_received.seq_no + 1
            if expected_seq_no != batch.seq_no:
                raise ValueError(
                    f"Expected sequence number {expected_seq_no} for batch from '{from_step}'"
                    f" to '{to_step}', but got {batch.seq_no}"
                )

        self._batches[to_step][from_step].append(batch)

        if self._step_input_batches_received(to_step):
            if to_step in self._accumulate_batches_for_steps:
                batches = list(self._batches[to_step].values())
                return _Batch.accumulate(to_step, batches)

            # Build a batch with the first batch from each predecessor
            batches = [batches.pop(0) for batches in self._batches[to_step].values()]
            return _Batch.from_batches(to_step, batches)

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
        accumulate_batches_for_steps = []
        for step_name in dag:
            step: "Step" = dag.get_step(step_name)["step"]

            # Skip generator steps as they don't have predecessors
            if step.is_generator:
                continue

            # For global steps, accumulate received batches until last one is received
            if step.is_global:
                accumulate_batches_for_steps.append(step_name)

            batches[step_name] = {}
            for predecessor in dag.get_step_predecessors(step_name):
                batches[step_name][predecessor] = []
        return cls(batches, accumulate_batches_for_steps)

    def _step_input_batches_received(self, step_name: str) -> bool:
        """Check if all the input batches for a step have been received. If the step is
        configured to accumulate batches, then it will check if it has received the last
        batch from all it's predecessors. Otherwise, it will check if it has received at
        least one batch from all it's predecessors.

        Args:
            step_name: The name of the step.

        Returns:
            A boolean indicating if all the inputs for the step have been received.
        """
        if step_name in self._accumulate_batches_for_steps:
            return all(
                batches[-1].last_batch for batches in self._batches[step_name].values()
            )

        return all(len(batches) > 0 for batches in self._batches[step_name].values())
