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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, TypedDict, Union

from typing_extensions import Self

from distilabel import __version__
from distilabel.pipeline._dag import DAG
from distilabel.pipeline.logging import get_logger
from distilabel.utils.serialization import TYPE_INFO_KEY, _Serializable

if TYPE_CHECKING:
    from os import PathLike

    from distilabel.steps.base import _Step


BASE_CACHE_DIR = Path.home() / ".cache" / "distilabel" / "pipelines"


class CacheFilenames(TypedDict):
    """Dictionary to store the filenames of a cached pipeline."""

    pipeline: Path
    batch_manager: Path
    data: Path


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
        self._logger = get_logger("pipeline")
        # It's set to None here, will be created in the call to run
        self._batch_manager: Optional["_BatchManager"] = None

    def __enter__(self) -> Self:
        """Set the global pipeline instance when entering a pipeline context."""
        _GlobalPipelineManager.set_pipeline(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Unset the global pipeline instance when exiting a pipeline context."""
        _GlobalPipelineManager.set_pipeline(None)
        self._cache()

    def _create_signature(self) -> str:
        """Makes a signature (hash) of a pipeline, using the step ids and the adjacency between them.

        The main use is to find the pipeline in the cache folder.

        Returns:
            int: Signature of the pipeline.
        """
        hasher = hashlib.sha1()

        steps_info = []
        pipeline_dump = self.dump()["pipeline"]
        for step in pipeline_dump["steps"]:
            step_info = step["name"]
            for argument, value in sorted(step["step"].items()):
                if (
                    (argument == TYPE_INFO_KEY)
                    or (argument == "llm")
                    or (value is None)
                ):
                    # NOTE: Should we include the LLM info at this stage??
                    continue

                if isinstance(value, dict):
                    # input_mappings/output_mappings
                    step_info += "-".join(
                        [f"{str(k)}-{str(v)}" for k, v in value.items()]
                    )
                elif isinstance(value, (list, tuple)):
                    # runtime_parameters_info
                    step_info += "-".join([str(v) for v in value])
                elif isinstance(value, (int, str)):
                    # batch_size/name
                    step_info += str(value)
                else:
                    raise ValueError(
                        f"Field '{argument}' in step '{step['name']}' has type {type(value)}, explicitly cast the type to 'str'."
                    )

            steps_info.append(step_info)

        connections_info = [
            f"{c['from']}-{'-'.join(c['to'])}" for c in pipeline_dump["connections"]
        ]
        hasher.update(",".join(steps_info + connections_info).encode())

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
        self._load_from_cache()

    def _add_step(self, step: "_Step") -> None:
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
            step: "_Step" = self.dag.get_step(step_name)["step"]
            step._set_runtime_parameters(step_parameters)

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the DAG content to a dict.

        Args:
            obj (Any): Unused, just kept to match the signature of the parent method.
            kwargs (Any): Unused, just kept to match the signature of the parent method.

        Returns:
            Dict[str, Any]: Internal representation of the DAG from networkx in a serializable format.
        """
        return self.dag.dump()

    def dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Transforms the pipeline to it's dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representing the pipeline.
        """
        return {"pipeline": super().dump(), "distilabel": {"version": __version__}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasePipeline":
        """Create a Pipeline from a dict containing the serialized data.

        Note:
            It's intended for internal use.

        Args:
            data (Dict[str, Any]): Dictionary containing the serialized data from a Pipeline.

        Returns:
            BasePipeline: Pipeline recreated from the dictionary info.
        """

        with cls() as pipe:
            pipe.dag = DAG.from_dict(data["pipeline"])
        return pipe

    @property
    def _cache_filenames(self) -> CacheFilenames:
        """Dictionary containing the object and the filenames to load them back..

        Returns:
            Path: Filenames where the pipeline content will be serialized.
        """
        folder = self._cache_dir / self._create_signature()
        return {
            "pipeline": folder / "pipeline.yaml",
            "batch_manager": folder / "batch_manager.json",
            "data": folder / "data.jsonl",
        }

    def _cache(self) -> None:
        """Saves the `BasePipeline` using the `_cache_filename` (won't overwrite the file)."""
        if not self._cache_filenames["pipeline"].exists():
            self.save(
                path=self._cache_filenames["pipeline"],
                format=self._cache_filenames["pipeline"].suffix.replace(".", ""),
            )
            if self._batch_manager is not None:
                self._batch_manager.save(
                    self._cache_filenames["batch_manager"],
                    format=self._cache_filenames["batch_manager"].suffix.replace(
                        ".", ""
                    ),
                )

    def _load_from_cache(self) -> None:
        """Will try to load the `BasePipeline` from the cache dir if found, updating
        the internal `DAG` and `_BatchManager`.
        """
        # Store the _cache_filename in a variable to avoid it changing when refreshing
        # the dag
        cache_filenames = self._cache_filenames
        if cache_filenames["pipeline"].exists():
            # Refresh the DAG to avoid errors when it's created within a context manager
            # (it will check the steps aren't already defined for the DAG).
            self.dag = DAG()
            new_class = self.from_yaml(cache_filenames["pipeline"])
            # Update the internal dag and batch_manager
            self.dag.G = new_class.dag.G
            if cache_filenames["batch_manager"].exists():
                self._batch_manager = _BatchManager.from_json(
                    cache_filenames["batch_manager"]
                )
            self._logger.info("💾 Load pipeline from cache")


@dataclass
class _Batch(_Serializable):
    """Dataclass to represent a batch of data to be processed by a `_Step`.

    Attributes:
        seq_no: The sequence number of the batch.
        step_name: The name of the step that will process the batch.
        last_batch: A flag to indicate if the batch is the last one.
        data: The data to be processed.
        accumulated: A flag to indicate if the batch is accumulated.
    """

    seq_no: int
    step_name: str
    last_batch: bool
    data: List[List[Dict[str, Any]]] = field(default_factory=list, repr=False)
    accumulated: bool = False

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
        return cls(
            seq_no=0, step_name=step_name, last_batch=True, data=data, accumulated=True
        )

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_Batch` to a dictionary, using the `dataclass` helper function.

        Args:
            obj (Any): Unused, just kept to match the signature of the parent method.
            kwargs (Any): Additional arguments that are kept to match the signature of the parent method.

        Returns:
            Dict[str, Any]: Internal representation of the `_Batch`.
        """
        return asdict(self)


@dataclass
class _BatchManagerStep(_Serializable):
    """A class that will accumulate data for a step from the predecessors and create
    batches for the step to process when there is enough data.

    Attributes:
        step_name: The name of the step that will process the data.
        accumulate: A flag to indicate if the data should be accumulated and create a
            batch with all the data received from the predecessors instead of creating
            batches with the `input_batch_size`.
        input_batch_size: The size of the batch to be created for the step to process.
            If `None`, then `accumulate` must be `True`. Defaults to `None`.
        data: A dictionary with the predecessor step name as the key and a list of
            dictionaries (rows) as the value.
        seq_no: The sequence number of the next batch to be created. It will be
            incremented for each batch created.
        last_batch_received: A list with the names of the steps that sent the last
            batch of data.
    """

    step_name: str
    accumulate: bool
    input_batch_size: Union[int, None] = None
    data: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    seq_no: int = 0
    last_batch_received: List[str] = field(default_factory=list)

    def add_batch(self, batch: _Batch) -> None:
        """Add a batch of data from `batch.step_name` to the step. It will accumulate the
        data and keep track of the last batch received from the predecessors.

        Args:
            batch: The output batch of an step to be processed by the step.
        """
        from_step = batch.step_name
        self.data[from_step].extend(batch.data[0])
        if batch.last_batch:
            self.last_batch_received.append(from_step)

    def get_batches(self) -> Iterable[_Batch]:
        """Create a new batch of data for the step to process. It will return `None` if
        there is not enough data to create a batch.

        Returns:
            A `_Batch` instance if there is enough data to create a batch. Otherwise,
            `None`.
        """
        while self._ready_to_create_batch():
            yield _Batch(
                seq_no=self._get_seq_no(),
                step_name=self.step_name,
                last_batch=self._last_batch(),
                data=self._get_data(),
                accumulated=self.accumulate,
            )

    @classmethod
    def from_step(
        cls, step: "_Step", predecessors: Iterable[str]
    ) -> "_BatchManagerStep":
        """Creates a `_BatchManagerStep` instance from a `_Step` instance and its
        predecessors.

        Returns:
            A `_BatchManagerStep` instance.
        """
        return cls(
            step_name=step.name,
            accumulate=step.is_global,
            input_batch_size=getattr(step, "input_batch_size", None),
            data={predecessor: [] for predecessor in predecessors},
        )

    def _get_seq_no(self) -> int:
        """Gets the sequence number for the next batch to be created and increments it.

        Returns:
            The sequence number for the next batch to be created.
        """
        seq_no = self.seq_no
        self.seq_no += 1
        return seq_no

    def _get_data(self) -> List[List[Dict[str, Any]]]:
        """Gets the data needed to create a batch for the step to process. If the step is
        accumulating data, then it will return a list with all the data received from the
        predecessors. Otherwise, it will return a list of data with the `input_batch_size`
        for each predecessor. In addition, it will remove the data used to create the
        batch from the step's data.

        Returns:
            The list of data needed to create a batch for the step to process.
        """
        if self.accumulate:
            data = list(self.data.values())
            self.data = {step_name: [] for step_name in self.data}
            return data

        data = []
        for step_name in self.data:
            step_data = self.data[step_name]
            data_for_batch, self.data[step_name] = (
                step_data[: self.input_batch_size],
                step_data[self.input_batch_size :],
            )
            data.append(data_for_batch)
        return data

    def _ready_to_create_batch(self) -> bool:
        """Checks if there is enough data to create a batch for the step. If the step is
        accumulating data, then it will return `True` if the last batch was received from
        all the predecessors. Otherwise, it will return `True` if there is enough data to
        create a batch for the step based on the `input_batch_size`.

        Returns:
            `True` if there is enough data to create a batch for the step. Otherwise,
            `False`.
        """
        if self.accumulate:
            return all(
                step in self.last_batch_received and len(rows) > 0
                for step, rows in self.data.items()
            )

        for step_name, rows in self.data.items():
            # If there are now rows but the last batch was already received, then there
            # are no more batch to be created
            if len(rows) == 0 and step_name in self.last_batch_received:
                return False

            # If there are not enough rows and the last batch was not received yet, then
            # there is not enough data yet to creata a batch
            if (
                self.input_batch_size
                and len(rows) < self.input_batch_size
                and step_name not in self.last_batch_received
            ):
                return False

        return True

    def _last_batch(self) -> bool:
        """Checks if the batch to be created is the last one i.e. if the last batch was
        received from all the predecessors.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        if self.accumulate:
            return all(step in self.last_batch_received for step in self.data.keys())

        for step_name, rows in self.data.items():
            if step_name not in self.last_batch_received:
                return False

            if (
                self.input_batch_size
                and len(rows) > self.input_batch_size
                and step_name in self.last_batch_received
            ):
                return False

        return True

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_BatchManagerStep` to a dictionary, using the `dataclass` helper function.

        Args:
            obj (Any): Unused, just kept to match the signature of the parent method.
            kwargs (Any): Additional arguments that are kept to match the signature of the parent method.

        Returns:
            Dict[str, Any]: Internal representation of the `_BatchManagerStep`.
        """
        return asdict(self)


class _BatchManager(_Serializable):
    """Class to manage the batches received from the steps. It keeps track of the
    received batches and returns new batches for the steps to process based on their
    input batch size and the batches received from the predecessors.

    Attributes:
        _steps: A dictionary with the step name as the key and a `_BatchManagerStep`
            instance as the value.
    """

    def __init__(self, steps: Dict[str, _BatchManagerStep]) -> None:
        """Initialize the `_BatchManager` instance.

        Args:
            steps: A dictionary with the step name as the key and a dictionary with the
                predecessor step name as the key and a list of batches as the value.
        """
        self._steps = steps

    def add_batch(self, to_step: str, batch: _Batch) -> Iterable[_Batch]:
        """Add an output batch from `batch.step_name` to `to_step`. If there is enough
        data for creating a `_Batch` for `to_step`, then it will return the batch to be
        processed. Otherwise, it will return `None`.

        Args:
            to_step: The name of the step that will process the batch.
            batch: The output batch of an step to be processed by `to_step`.

        Returns:
            If there is enough data for creating a batch for `to_step`, then it will return
            the batch to be processed. Otherwise, it will return `None`.

        Raises:
            ValueError: If `to_step` is not found in the batch manager.
        """
        if to_step not in self._steps:
            raise ValueError(f"Step '{to_step}' not found in the batch manager.")

        step = self._steps[to_step]
        step.add_batch(batch)
        yield from step.get_batches()

    @classmethod
    def from_dag(cls, dag: "DAG") -> "_BatchManager":
        """Create a `_BatchManager` instance from a `DAG` instance.

        Args:
            dag: The `DAG` instance.

        Returns:
            A `_BatchManager` instance.
        """
        steps = {}
        for step_name in dag:
            step: "_Step" = dag.get_step(step_name)["step"]
            if step.is_generator:
                continue
            batch_manager_step = _BatchManagerStep.from_step(
                step, dag.get_step_predecessors(step_name)
            )
            steps[step_name] = batch_manager_step
        return cls(steps)

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_BatchManager` to a dictionary.

        Args:
            obj (Any): Unused, just kept to match the signature of the parent method.
            kwargs (Any): Additional arguments that are kept to match the signature of the parent method.

        Returns:
            Dict[str, Any]: Internal representation of the `_BatchManager`.
        """
        return {name: step.dump() for name, step in self._steps.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_BatchManager":
        """Loads a `_BatchManager` from its serialized content in a dictionary.

        Args:
            data: The serialized batch manager.

        Returns:
            A `_BatchManager` instance.
        """
        # Remove the type info, we already know its a _BatchManager, and there aren't subclasses of it
        data.pop(TYPE_INFO_KEY)
        # Also there is only one type of _BatchManagerStep, so we can call it directly instead of generically
        # via _get_class
        return cls(
            {name: _BatchManagerStep.from_dict(step) for name, step in data.items()}
        )
