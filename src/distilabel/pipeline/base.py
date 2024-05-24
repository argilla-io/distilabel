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

import copy
import hashlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
)

import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import Self

from distilabel import __version__
from distilabel.pipeline._dag import DAG
from distilabel.pipeline.constants import (
    RECEIVES_ROUTED_BATCHES_ATTR_NAME,
    ROUTING_BATCH_FUNCTION_ATTR_NAME,
    STEP_ATTR_NAME,
)
from distilabel.utils.files import list_files_in_dir
from distilabel.utils.serialization import (
    TYPE_INFO_KEY,
    _check_is_dir,
    _Serializable,
    read_json,
)

if TYPE_CHECKING:
    from os import PathLike

    from distilabel.distiset import Distiset
    from distilabel.pipeline.routing_batch_function import RoutingBatchFunction
    from distilabel.steps.base import _Step
    from distilabel.utils.serialization import StrOrPath


BASE_CACHE_DIR = Path.home() / ".cache" / "distilabel" / "pipelines"


class _CacheLocation(TypedDict):
    """Dictionary to store the filenames and directories of a cached pipeline.

    Attributes:
        pipeline: The filename where the pipeline content will be serialized.
        batch_manager: The filename where the batch manager content will be serialized.
        data: The directory where the output data of each leaf step will be stored.
        log_file: The filename where the logs will be stored.
    """

    pipeline: Path
    batch_manager: Path
    data: Path
    log_file: Path


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
        name: The name of the pipeline.
        description: A description of the pipeline.
        dag: The `DAG` instance that represents the pipeline.
        _cache_dir: The directory where the pipeline will be cached.
        _logger: The logger instance that will be used by the pipeline.
        _batch_manager: The batch manager that will manage the batches received from the
            steps while running the pipeline.
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        cache_dir: Optional["PathLike"] = None,
        enable_metadata: bool = False,
    ) -> None:
        """Initialize the `BasePipeline` instance.

        Args:
            name: The name of the pipeline.
            description: A description of the pipeline. Defaults to `None`.
            cache_dir: A directory where the pipeline will be cached. Defaults to `None`.
            enable_metadata: Whether to include the distilabel metadata column for the pipeline
                in the final `Distiset`. It contains metadata used by distilabel, for example
                the raw outputs of the `LLM` without processing would be here, inside `raw_output_...`
                field. Defaults to `False`.
        """
        self.name = name
        self.description = description
        self._enable_metadata = enable_metadata
        self.dag = DAG()

        if cache_dir:
            self._cache_dir = Path(cache_dir)
        elif env_cache_dir := os.getenv("DISTILABEL_CACHE_DIR"):
            self._cache_dir = Path(env_cache_dir)
        else:
            self._cache_dir = BASE_CACHE_DIR

        self._logger = logging.getLogger("distilabel.pipeline")

        # It's set to None here, will be created in the call to run
        self._batch_manager: Optional["_BatchManager"] = None
        self._dry_run: bool = False

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

        steps_info = []
        pipeline_dump = self.dump()["pipeline"]

        for step in pipeline_dump["steps"]:
            step_info = step["name"]
            for argument, value in sorted(step[STEP_ATTR_NAME].items()):
                if (argument == TYPE_INFO_KEY) or (value is None):
                    continue

                if isinstance(value, dict):
                    # input_mappings/output_mappings
                    step_info += "-".join(
                        [f"{str(k)}-{str(v)}" for k, v in value.items()]
                    )
                elif isinstance(value, (list, tuple)):
                    # runtime_parameters_info
                    step_info += "-".join([str(v) for v in value])
                elif isinstance(value, (int, str, float)):
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

        routing_batch_functions_info = []
        for function in pipeline_dump["routing_batch_functions"]:
            step = function["step"]
            routing_batch_function: "RoutingBatchFunction" = self.dag.get_step(step)[
                ROUTING_BATCH_FUNCTION_ATTR_NAME
            ]
            if type_info := routing_batch_function._get_type_info():
                step += f"-{type_info}"

        hasher.update(
            ",".join(
                steps_info + connections_info + routing_batch_functions_info
            ).encode()
        )

        return hasher.hexdigest()

    def run(
        self,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache: bool = True,
    ) -> "Distiset":  # type: ignore
        """Run the pipeline. It will set the runtime parameters for the steps and validate
        the pipeline.

        This method should be extended by the specific pipeline implementation,
        adding the logic to run the pipeline.

        Args:
            parameters: A dictionary with the step name as the key and a dictionary with
                the runtime parameters for the step as the value. Defaults to `None`.
            use_cache: Whether to use the cache from previous pipeline runs. Defaults to
                `True`.

        Returns:
            The `Distiset` created by the pipeline.
        """
        if use_cache:
            self._load_from_cache()
        self._set_runtime_parameters(parameters or {})
        self.dag.validate()

    def dry_run(
        self,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 1,
    ) -> "Distiset":
        """Do a dry run to test the pipeline runs as expected.

        Running a `Pipeline` in dry run mode will set all the `batch_size` of generator steps
        to the specified `batch_size`, and run just with a single batch, effectively
        running the whole pipeline with a single example. The cache will be set to `False`.

        Args:
            parameters: A dictionary with the step name as the key and a dictionary with
                the runtime parameters for the step as the value. Defaults to `None`.
            batch_size: The batch size of the unique batch generated by the generators
                steps of the pipeline. Defaults to `1`.

        Returns:
            Will return the `Distiset` as the main run method would do.
        """
        self._dry_run = True

        for step_name in self.dag:
            step = self.dag.get_step(step_name)[STEP_ATTR_NAME]

            if step.is_generator:
                if not parameters:
                    parameters = {}
                parameters[step_name] = {"batch_size": batch_size}

        distiset = self.run(parameters=parameters, use_cache=False)

        self._dry_run = False
        return distiset

    def get_runtime_parameters_info(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the runtime parameters for the steps in the pipeline.

        Returns:
            A dictionary with the step name as the key and a list of dictionaries with
            the parameter name and the parameter info as the value.
        """
        runtime_parameters = {}
        for step_name in self.dag:
            step: "_Step" = self.dag.get_step(step_name)[STEP_ATTR_NAME]
            runtime_parameters[step_name] = step.get_runtime_parameters_info()
        return runtime_parameters

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

        # Check if `from_step` has a `routing_batch_function`. If it does, then mark
        # `to_step` as a step that will receive a routed batch.
        node = self.dag.get_step(from_step)  # type: ignore
        routing_batch_function = node.get(ROUTING_BATCH_FUNCTION_ATTR_NAME, None)
        self.dag.set_step_attr(
            name=to_step,
            attr=RECEIVES_ROUTED_BATCHES_ATTR_NAME,
            value=routing_batch_function is not None,
        )

    def _add_routing_batch_function(
        self, step_name: str, routing_batch_function: "RoutingBatchFunction"
    ) -> None:
        """Add a routing batch function to a step.

        Args:
            step_name: The name of the step that will receive the routed batch.
            routing_batch_function: The function that will route the batch to the step.
        """
        self.dag.set_step_attr(
            name=step_name,
            attr=ROUTING_BATCH_FUNCTION_ATTR_NAME,
            value=routing_batch_function,
        )

    def _set_runtime_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> None:
        """Set the runtime parameters for the steps in the pipeline.

        Args:
            parameters: A dictionary with the step name as the key and a dictionary with
            the parameter name as the key and the parameter value as the value.
        """
        step_names = set(self.dag.G)
        for step_name, step_parameters in parameters.items():
            if step_name not in step_names:
                self._logger.warning(
                    f"❓ Step '{step_name}' provided in `Pipeline.run(parameters={{...}})` not found in the pipeline."
                    f" Available steps are: {step_names}."
                )
            else:
                step: "_Step" = self.dag.get_step(step_name)[STEP_ATTR_NAME]
                step.set_runtime_parameters(step_parameters)

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
        return {
            "distilabel": {"version": __version__},
            "pipeline": {
                "name": self.name,
                "description": self.description,
                **super().dump(),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create a Pipeline from a dict containing the serialized data.

        Note:
            It's intended for internal use.

        Args:
            data (Dict[str, Any]): Dictionary containing the serialized data from a Pipeline.

        Returns:
            BasePipeline: Pipeline recreated from the dictionary info.
        """
        name = data["pipeline"]["name"]
        description = data["pipeline"].get("description")
        with cls(name=name, description=description) as pipe:
            pipe.dag = DAG.from_dict(data["pipeline"])
        return pipe

    @property
    def _cache_location(self) -> _CacheLocation:
        """Dictionary containing the the object that will stored and the location,
        whether it is a filename or a folder.

        Returns:
            Path: Filenames where the pipeline content will be serialized.
        """
        folder = self._cache_dir / self.name / self._create_signature()
        return {
            "pipeline": folder / "pipeline.yaml",
            "batch_manager": folder / "batch_manager.json",
            "data": folder / "data",
            "log_file": folder / "pipeline.log",
        }

    def _cache(self) -> None:
        """Saves the `BasePipeline` using the `_cache_filename`."""
        self.save(
            path=self._cache_location["pipeline"],
            format=self._cache_location["pipeline"].suffix.replace(".", ""),  # type: ignore
        )
        if self._batch_manager is not None:
            self._batch_manager.cache(self._cache_location["batch_manager"])
        self._logger.debug("Pipeline and batch manager saved to cache.")

    def _load_from_cache(self) -> None:
        """Will try to load the `BasePipeline` from the cache dir if found, updating
        the internal `DAG` and `_BatchManager`.
        """
        cache_loc = self._cache_location
        if cache_loc["pipeline"].exists():
            if cache_loc["batch_manager"].exists():
                self._batch_manager = _BatchManager.load_from_cache(
                    cache_loc["batch_manager"]
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
        created_from: A dictionary containing the `seq_no` of the batches of the steps that
            were used to create this batch.
        size: The size of the batch.
    """

    seq_no: int
    step_name: str
    last_batch: bool
    data: List[List[Dict[str, Any]]] = field(default_factory=list, repr=False)
    data_hash: Optional[str] = None
    accumulated: bool = False
    created_from: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    batch_routed_to: List[str] = field(default_factory=list)
    size: int = 0

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

    def set_data(self, data: List[List[Dict[str, Any]]]) -> None:
        """Sets the data of the batch and updates the size of the batch.

        Args:
            data: The data of the batch.
        """
        self.data = data
        self.size = len(data[0])
        self._update_data_hash()

    def get_data(self, num_rows: Union[int, None] = None) -> List[Dict[str, Any]]:
        """Takes `num_rows` from the data of the batch and returns it. This method will
        also remove the data from the batch and update the hash of the data.

        Args:
            num_rows: The number of rows to take from the data. If `None`, then all the
                data will be taken. Defaults to `None`.

        Returns:
            A list with the data taken from the batch.
        """

        if num_rows is None:
            data = self.data[0]
            self.data = []
        else:
            data = self.data[0][:num_rows]
            self.data[0] = self.data[0][num_rows:]

        self._update_data_hash()
        return data

    def _update_data_hash(self) -> None:
        """Updates the hash of the data of the batch."""
        self.data_hash = hashlib.sha1(str(self.data).encode()).hexdigest()

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
            obj: Unused, just kept to match the signature of the parent method.
            kwargs: Additional arguments that are kept to match the signature of the parent method.

        Returns:
            A `dict` containing the internal representation of the `_Batch`.
        """

        include_batch_data = kwargs.get("include_batch_data", True)

        dump = {
            "seq_no": self.seq_no,
            "step_name": self.step_name,
            "last_batch": self.last_batch,
            "data_hash": self.data_hash,
            "accumulated": self.accumulated,
            "created_from": self.created_from,
            "batch_routed_to": self.batch_routed_to,
            "size": self.size,
        }

        if include_batch_data:
            dump["data"] = self.data

        return dump

    def copy(self) -> "_Batch":
        """Creates a copy of the `_Batch` instance.

        Returns:
            A copy of the `_Batch` instance.
        """
        return copy.deepcopy(self)


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
        convergence_step: A flag to indicate if the step is a convergence step. An
            `Step` is a convergence step if all its predecessors are receiving routed
            batches. Defaults to `False`.
        convergence_step_batches_consumed: A dictionary in which the key is the `seq_no`
            of the batch created by step A, that was used by step B and C and obtained from
            the `created_from` of the batches created by them. It's used to know if all
            the batches from B and C steps created from batches of A have been consumed
            by D, in order to not mess up the order of the batches. Only used if `convergence_step=True`.
            Defaults to an empty dictionary.
        next_expected_created_from_batch_seq_no: The next expected sequence number of the
            batch from step A used by steps B and C and obtained from the `created_from`
            of the batches created by them. It's used to avoid messing up the order of the
            batches. Only used if `convergence_step=True`. Defaults to `0`.
    """

    step_name: str
    accumulate: bool
    input_batch_size: Union[int, None] = None
    data: Dict[str, List[_Batch]] = field(default_factory=dict)
    seq_no: int = 0
    last_batch_received: List[str] = field(default_factory=list)
    convergence_step: bool = False
    convergence_step_batches_consumed: Dict[int, Dict[str, int]] = field(
        default_factory=dict
    )
    next_expected_created_from_batch_seq_no: int = 0

    def add_batch(self, batch: _Batch, prepend: bool = False) -> None:
        """Add a batch of data from `batch.step_name` to the step. It will accumulate the
        data and keep track of the last batch received from the predecessors.

        Args:
            batch: The output batch of an step to be processed by the step.
            prepend: If `True`, the content of the batch will be added at the start of
                the buffer.
        """
        from_step = batch.step_name

        if prepend:
            self.data[from_step].insert(0, batch)
        else:
            self.data[from_step].append(batch)

        if batch.last_batch:
            self.last_batch_received.append(from_step)

    def get_batch(self) -> Union[_Batch, None]:
        """Create a new batch of data for the step to process. It will return `None` if
        there is not enough data to create a batch.

        Returns:
            A `_Batch` instance if there is enough data to create a batch. Otherwise,
            `None`.
        """
        if not self._ready_to_create_batch():
            return None

        # `_last_batch` must be called before `_get_data`, as `_get_data` will update the
        # list of data which is used to determine if the batch to be created is the last one.
        # TODO: remove `_last_batch` method and integrate logic in `_get_data`
        last_batch = self._last_batch()
        data, created_from, batch_routed_to = self._get_data()

        return _Batch(
            seq_no=self._get_seq_no(),
            step_name=self.step_name,
            last_batch=last_batch,
            data=data,
            accumulated=self.accumulate,
            created_from=created_from,
            batch_routed_to=batch_routed_to,
        )

    def empty_buffers(self) -> List[str]:
        """Checks if the input buffer for the step is empty.

        Returns:
            The name of the previous steps for which the input buffer for this step is
            empty.
        """
        if self.accumulate:
            return [
                previous_step
                for previous_step in self.data.keys()
                if previous_step not in self.last_batch_received
            ]

        return [
            previous_step
            for previous_step, batches in self.data.items()
            if previous_step not in self.last_batch_received
            and sum(len(batch.data[0]) for batch in batches) < self.input_batch_size  # type: ignore
        ]

    @classmethod
    def from_step(
        cls, step: "_Step", predecessors: Iterable[str], convergence_step: bool = False
    ) -> "_BatchManagerStep":
        """Creates a `_BatchManagerStep` instance from a `_Step` instance and its
        predecessors.

        Args:
            step: The `_Step` instance.
            predecessors: The names of the predecessors of the step.
            convergence_step: A flag to indicate if the step is a convergence step. An
                `Step` is a convergence step if all its predecessors are receiving routed
                batches. Defaults to `False`.

        Returns:
            A `_BatchManagerStep` instance.
        """
        return cls(
            step_name=step.name,  # type: ignore
            accumulate=step.is_global,
            input_batch_size=getattr(step, "input_batch_size", None),
            data={predecessor: [] for predecessor in predecessors},
            convergence_step=convergence_step,
        )

    def _get_seq_no(self) -> int:
        """Gets the sequence number for the next batch to be created and increments it.

        Returns:
            The sequence number for the next batch to be created.
        """
        seq_no = self.seq_no
        self.seq_no += 1
        return seq_no

    def _get_data(
        self,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int]]], List[str]]:
        """Gets the data needed to create a batch for the step to process. If the step is
        accumulating data, then it will return a list with all the data received from the
        predecessors. Otherwise, it will return a list of data with the `input_batch_size`
        for each predecessor. In addition, it will remove the data used to create the
        batch from the step's data.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process, a dictionary with the sequence numbers of the batches that were used
            to create the batch and the list of steps to which the batch was routed to if
            the step is a normal step.
        """
        if self.accumulate:
            # Steps accumulating cannot receive routed batches
            return self._get_data_for_accumulate() + ([],)

        if self.convergence_step:
            # Convergence steps will receive routed batches, but we need to clean the
            # `batch_routed_to` list
            return self._get_data_for_convergence_step() + ([],)

        return self._get_data_normal()

    def _get_data_for_accumulate(
        self,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int]]]]:
        """Gets the data needed to create a batch for the step to process when the step
        is accumulating data. It will return a list with all the data received from the
        predecessors. In addition, it will remove the data used to create the batch from
        the step's data.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process and a dictionary with the sequence numbers of the batches that were
            used to create the batch.
        """
        data = []
        batches_used = {}
        for step_name, batches in self.data.items():
            batches_used[step_name] = []
            for batch in batches:
                batches_used[step_name].append((batch.seq_no, batch.size))
            data.append([row for batch in batches for row in batch.get_data()])
        # Reset the data buffer
        self.data = {step_name: [] for step_name in self.data}
        return data, batches_used

    def _get_data_for_convergence_step(
        self,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int]]]]:
        """Gets the data needed to create a batch for the step to process when the step is
        a convergence step.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process and a dictionary with the sequence numbers of the batches that were
            used to create the batch.
        """
        grouped_batches = self._group_batches_by_created_from()
        seq_no, batches = grouped_batches[0]

        remaining_rows_per_step: Dict[str, int] = {
            step_name: self.input_batch_size
            for step_name in self.data  # type: ignore
        }
        batches_used = defaultdict(list)
        data = defaultdict(list)
        for batch, batch_size in batches:
            remaining_rows = remaining_rows_per_step[batch.step_name]
            selected_data = batch.get_data(remaining_rows)
            data[batch.step_name].extend(selected_data)

            # If A -> [B, C] -> D, then in D (this step) we keep track of the remaining
            # rows from the batches of A that B and C used to create the `batches`.
            batch_size = self.convergence_step_batches_consumed.setdefault(
                seq_no, {}
            ).get(batch.step_name, batch_size)
            remaining_rows_in_batch = batch_size - len(selected_data)
            self.convergence_step_batches_consumed[seq_no].update(
                {batch.step_name: remaining_rows_in_batch}
            )

            # Update the remaining rows
            num_rows = len(selected_data)
            remaining_rows_per_step[batch.step_name] -= num_rows  # type: ignore

            # Keep track of the batches used to create the batch
            batches_used[batch.step_name].append((batch.seq_no, batch.size))

            # If the batch was entirely consumed, then remove it from the buffer
            if len(batch.data[0]) == 0:
                self.data[batch.step_name].remove(batch)
                continue

        # If all the batches grouped by the `seq_no` in `created_from` were consumed, then
        # we can update the `next_expected_created_from_batch_seq_no` to the next one
        # to avoid skipping batches.
        no_remaining_rows = all(
            count == 0
            for count in self.convergence_step_batches_consumed[seq_no].values()
        )
        if no_remaining_rows:
            self.next_expected_created_from_batch_seq_no += 1

        return list(data.values()), dict(batches_used)

    def _get_data_normal(
        self,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int]]], List[str]]:
        """Gets the data needed to create a batch for the step to process when the step is
        not accumulating data. It will return a list of data with the `input_batch_size`
        for each predecessor. In addition, it will remove the data used to create the batch
        from the step's data.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process, a dictionary with the sequence numbers of the batches that were used
            to create the batch and the list of steps to which the batch was routed to if
            the step is a convergence step.
        """
        data = []
        batches_used = defaultdict(list)
        batch_routed_to = []
        for step_name in self.data:
            # For each step batches buffer, we will create a batch with the `input_batch_size`
            # using the data from the buffer. We will remove the consumed batches (no data
            # left) and update the batch data with the remaining data.
            step_data = []
            idx_drop_batches = []
            remaining_rows: int = self.input_batch_size  # type: ignore
            for idx, batch in enumerate(self.data[step_name]):
                if remaining_rows == 0:
                    break

                # Get `remaining_rows` or the remaining rows in the batch and add it to
                # the step data that will be used to create the batch
                selected_data = batch.get_data(remaining_rows)
                step_data.extend(selected_data)
                batch_routed_to = batch.batch_routed_to

                # Update the remaining rows
                num_rows = len(selected_data)
                remaining_rows -= num_rows

                # Keep track of the batches used to create the batch
                batches_used[step_name].append((batch.seq_no, batch.size))

                # If the batch was entirely consumed, then remove it from the buffer
                if len(batch.data[0]) == 0:
                    idx_drop_batches.append(idx)
                    continue

            # Remove the batches that were entirely consumed
            idx_drop_batches.reverse()
            for idx in idx_drop_batches:
                self.data[step_name].pop(idx)

            data.append(step_data)

        return data, dict(batches_used), batch_routed_to

    def _ready_to_create_batch(self) -> bool:
        """Checks if there is enough data to create a batch for the step.

        Returns:
            `True` if there is enough data to create a batch for the step. Otherwise,
            `False`.
        """
        if self.accumulate:
            return self._ready_to_create_batch_accumulate()

        if self.convergence_step:
            return self._ready_to_create_batch_convergence_step()

        return self._ready_to_create_batch_normal()

    def _ready_to_create_batch_accumulate(self) -> bool:
        """Checks if there is enough data for an step accumulating data. It will return
        `True` if the last batch was received from all the predecessors.

        Returns:
            `True` if ready to create a batch, `False` otherwise.
        """
        return all(
            step in self.last_batch_received
            and sum(len(batch.data[0]) for batch in batches) >= 0
            for step, batches in self.data.items()
        )

    def _ready_to_create_batch_convergence_step(self) -> bool:
        """Checks if there is enough data for creating a batch for an step in which output
        batches that were generated by steps that received routed batches are received.
        It will return `True`, if all the output batches that were generated from a routed
        batch have been received.

        Returns:
            `True` if ready to create a batch, `False` otherwise.
        """
        grouped_batches = self._group_batches_by_created_from()
        if not grouped_batches:
            return False
        seq_no, batches = grouped_batches[0]

        # If the `seq_no` from the `created_from` field is not the expected one, then
        # we cannot create a batch yet or the order will be messed up
        if seq_no != self.next_expected_created_from_batch_seq_no:
            return False

        # Not all output batches to which the input batch was routed to haven't been
        # received
        batch_routed_to = batches[0][0].batch_routed_to
        batches_received_from = {batch.step_name for batch, _ in batches}
        if any(step_name not in batches_received_from for step_name in batch_routed_to):
            return False

        # There are output batches to which the input batch was routed to from all
        # the steps. Check if there is enough data for creating a batch with `input_batch_size`
        rows_per_step = defaultdict(lambda: 0)
        for batch, _ in batches:
            num_rows = len(batch.data[0])
            rows_per_step[batch.step_name] += num_rows

        # If there aren't at least `input_batch_size` rows from each step, then there
        # isn't enough data to create a batch
        if not all(
            num_rows >= self.input_batch_size or step_name in self.last_batch_received  # type: ignore
            for step_name, num_rows in rows_per_step.items()
        ):
            return False

        return True

    def _ready_to_create_batch_normal(self) -> bool:
        """Checks if there is enough data for creating a batch for a normal step. It will
        be `True` it there are at least `input_batch_size` rows from each predecessor step.

        Returns:
            `True` if ready to create a batch, `False` otherwise.
        """
        for step_name, batches in self.data.items():
            num_rows = sum(len(batch.data[0]) for batch in batches)

            # If there are now rows but the last batch was already received, then there
            # are no more batch to be created
            if num_rows == 0 and step_name in self.last_batch_received:
                return False

            # If there are not enough rows and the last batch was not received yet, then
            # there is not enough data yet to creata a batch
            if (
                self.input_batch_size
                and num_rows < self.input_batch_size
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
            return self._last_batch_accumulate()

        if self.convergence_step:
            return self._last_batch_convergence_step()

        return self._last_batch_normal()

    def _last_batch_accumulate(self) -> bool:
        """Checks if the batch to be created is the last one for an step accumulating data.
        `True` if the last batch was received from all the predecessors.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        return all(step in self.last_batch_received for step in self.data.keys())

    def _last_batch_convergence_step(self) -> bool:
        """Checks if the batch to be created is the last one for a convergence step. `True`
        if the last batch of all the steps (`batch_routed_to`) in the last routed batch
        have been received.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        grouped_batches = self._group_batches_by_created_from()
        if not grouped_batches:
            return False
        _, batches = grouped_batches[0]

        for batch, _ in batches:
            if not batch.last_batch:
                return False

            if len(batch.data[0]) > self.input_batch_size:  # type: ignore
                return False

        return True

    def _last_batch_normal(self) -> bool:
        """Checks if the batch to be created is the last one for a normal step. `True` if
        there is no more data to be received from the predecessors.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        for step_name, batches in self.data.items():
            if step_name not in self.last_batch_received:
                return False

            num_rows = sum(len(batch.data[0]) for batch in batches)

            if (
                self.input_batch_size
                and num_rows > self.input_batch_size
                and step_name in self.last_batch_received
            ):
                return False

        return True

    def _group_batches_by_created_from(
        self,
    ) -> List[Tuple[int, List[Tuple["_Batch", int]]]]:
        """Group the batches by the first key of `created_from` field. This method is
        meant to be used only with a `convergence_step`.

        Returns:
            A list of the batches grouped by the `seq_no` of the first step name in `created_from`.
            The list is sorted by the `seq_no`.
        """
        grouped_batches: Dict[int, List[Tuple["_Batch", int]]] = defaultdict(list)
        for batches in self.data.values():
            for batch in batches:
                first_key = next(iter(batch.created_from))
                batch_seq_no, batch_size = batch.created_from[first_key][0]
                grouped_batches[batch_seq_no].append((batch, batch_size))
        return sorted((seq_no, batches) for seq_no, batches in grouped_batches.items())

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_BatchManagerStep` to a dictionary, using the `dataclass` helper function.

        Args:
            obj: Unused, just kept to match the signature of the parent method.
            kwargs: Additional arguments that are kept to match the signature of the parent method.

        Returns:
            Internal representation of the `_BatchManagerStep`.
        """
        return {
            "step_name": self.step_name,
            "accumulate": self.accumulate,
            "input_batch_size": self.input_batch_size,
            "data": {
                step_name: [batch.dump(**kwargs) for batch in batches]
                for step_name, batches in self.data.items()
            },
            "seq_no": self.seq_no,
            "last_batch_received": self.last_batch_received,
            "convergence_step": self.convergence_step,
            "convergence_step_batches_consumed": self.convergence_step_batches_consumed,
            "next_expected_created_from_batch_seq_no": self.next_expected_created_from_batch_seq_no,
        }


LAST_BATCH_SENT_FLAG = "last_batch_sent"


class _BatchManager(_Serializable):
    """Class to manage the batches received from the steps. It keeps track of the
    received batches and returns new batches for the steps to process based on their
    input batch size and the batches received from the predecessors.

    Attributes:
        steps: A dictionary with the step name as the key and a `_BatchManagerStep`
            instance as the value.
        last_batch_received: A dictionary with the step name as the key and a flag to
            indicate whether we received the last batch from the step.
    """

    def __init__(
        self,
        steps: Dict[str, _BatchManagerStep],
        last_batch_received: Dict[str, Union[_Batch, None]],
        last_batch_sent: Dict[str, Union[_Batch, None]],
        last_batch_flag_sent_to: List[str],
    ) -> None:
        """Initialize the `_BatchManager` instance.

        Args:
            steps: A dictionary with the step name as the key and a dictionary with the
                predecessor step name as the key and a list of batches as the value.
            last_batch_received: A dictionary with the step name as the key and a the last
                `_Batch` received from the step.
            last_batch_sent: A dictionary with the step name as the key and a the last
                `_Batch` sent to the step.
            last_batch_flag_sent_to: A list with the names of the steps to which `LAST_BATCH_SENT_FLAG`
                was sent.
        """

        self._steps = steps
        self._last_batch_received = last_batch_received
        self._last_batch_sent = last_batch_sent
        self._last_batch_flag_sent_to = last_batch_flag_sent_to

    def can_generate(self) -> bool:
        """Checks if there are still batches to be processed by the steps.

        Returns:
            `True` if there are still batches to be processed by the steps. Otherwise,
            `False`.
        """

        for step_name, batch in self._last_batch_received.items():
            if step_name not in self._last_batch_flag_sent_to:
                if not batch:
                    return True

                if not batch.last_batch:
                    return True

                if not self.get_last_batch_sent(step_name):
                    return True

        return False

    def register_batch(self, batch: _Batch) -> None:
        """Method to register a batch received from a step. It will keep track of the
        sequence number and the last batch received from the step in the internal maps.

        Args:
            batch: _Batch from which we will register the sequence number and the last batch received.
        """
        self._last_batch_received[batch.step_name] = batch

    def get_last_batch(self, step_name: str) -> Union[_Batch, None]:
        """Gets the last batch received from a step.

        Args:
            step_name: The name of the step.

        Returns:
            The last batch received from the step or `None` if no batch was received.
        """
        return self._last_batch_received.get(step_name)

    def add_batch(self, to_step: str, batch: _Batch, prepend: bool = False) -> None:
        """Add an output batch from `batch.step_name` to `to_step`.

        Args:
            to_step: The name of the step that will process the batch.
            batch: The output batch of an step to be processed by `to_step`.
            prepend: If `True`, the content of the batch will be added at the start of
                the buffer.

        Raises:
            ValueError: If `to_step` is not found in the batch manager.
        """
        if to_step not in self._steps:
            raise ValueError(f"Step '{to_step}' not found in the batch manager.")

        step = self._steps[to_step]
        step.add_batch(batch, prepend)

    def get_batch(self, step_name: str) -> Union[_Batch, None]:
        """Get the next batch to be processed by the step.

        Args:
            step_name: The name of the step that will process the batch.

        Returns:
            A `_Batch` instance if there is a batch to be processed by the step. Otherwise,
            `None`.
        """
        if step_name not in self._steps:
            raise ValueError(f"Step '{step_name}' not found in the batch manager.")

        return self._steps[step_name].get_batch()

    def step_empty_buffers(self, step_name: str) -> List[str]:
        """Checks if the input buffer for a step is empty.

        Args:
            step_name: The name of the step.

        Returns:
            The name of the previous steps for which the input buffer for this step is
            empty.
        """
        return self._steps[step_name].empty_buffers()

    def set_last_batch_sent(self, batch: "_Batch") -> None:
        """Set the last batch sent to a step.

        Args:
            batch: The last batch sent to a step.
        """
        self._last_batch_sent[batch.step_name] = batch

    def get_last_batch_sent(self, step_name: str) -> Union["_Batch", None]:
        """Get the last batch sent to a step.

        Args:
            step_name: The name of the step.

        Returns:
            The last batch sent to a step or `None` if no batch was sent.
        """
        return self._last_batch_sent.get(step_name, None)

    def set_last_batch_flag_sent_to(self, step_name: str) -> None:
        """Set the flag to indicate that the last batch was sent to a step.

        Args:
            step_name: The name of the step.
        """
        self._last_batch_flag_sent_to.append(step_name)

    @classmethod
    def from_dag(cls, dag: "DAG") -> "_BatchManager":
        """Create a `_BatchManager` instance from a `DAG` instance.

        Args:
            dag: The `DAG` instance.

        Returns:
            A `_BatchManager` instance.
        """
        steps = {}
        last_batch_received = {}
        last_batch_sent = {}
        for step_name in dag:
            step: "_Step" = dag.get_step(step_name)[STEP_ATTR_NAME]
            last_batch_received[step.name] = None
            last_batch_sent[step.name] = None
            if step.is_generator:
                continue
            predecessors = list(dag.get_step_predecessors(step_name))
            convergence_step = all(
                dag.get_step(predecessor).get(RECEIVES_ROUTED_BATCHES_ATTR_NAME, False)
                for predecessor in predecessors
            )
            batch_manager_step = _BatchManagerStep.from_step(
                step=step,
                predecessors=predecessors,
                convergence_step=convergence_step,
            )
            steps[step_name] = batch_manager_step
        return cls(steps, last_batch_received, last_batch_sent, [])

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_BatchManager` to a dictionary.

        Args:
            obj (Any): Unused, just kept to match the signature of the parent method.
            kwargs (Any): Additional arguments that are kept to match the signature of the parent method.

        Returns:
            Dict[str, Any]: Internal representation of the `_BatchManager`.
        """
        return {
            "steps": {name: step.dump(**kwargs) for name, step in self._steps.items()},
            "last_batch_received": {
                step_name: batch.dump(**kwargs) if batch is not None else None
                for step_name, batch in self._last_batch_received.items()
            },
            "last_batch_sent": {
                step_name: batch.dump(**kwargs) if batch is not None else None
                for step_name, batch in self._last_batch_sent.items()
            },
            "last_batch_flag_sent_to": self._last_batch_flag_sent_to,
        }

    def cache(self, path: "StrOrPath") -> None:
        """Cache the `_BatchManager` to a file.

        Args:
            path: The path to the file where the `_BatchManager` will be cached. If `None`,
                then the `_BatchManager` will be cached in the default cache folder.
        """
        path = Path(path)

        # Do not include `_Batch` data so `dump` is fast
        dump = self.dump(include_batch_data=False)
        batch_manager_step_files = {}

        # Do this to avoid modifying the dictionary while iterating over it
        batch_manager_steps = set(dump["steps"].keys())
        for step_name in batch_manager_steps:
            step_dump = dump["steps"].pop(step_name)

            # Create a directory for each batch manager step to store their batches
            batch_manager_step_dir = path.parent / f"batch_manager_steps/{step_name}"
            batch_manager_step_dir.mkdir(parents=True, exist_ok=True)

            # Store each `_BatchManagerStep` `_Batch`es in a separete file
            for buffered_step_name in step_dump["data"]:
                step_batches_dir = batch_manager_step_dir / buffered_step_name
                step_batches_dir.mkdir(parents=True, exist_ok=True)

                # Store each `_Batch` in a separate file
                keep_batches = []
                for batch_dump in step_dump["data"][buffered_step_name]:
                    # Generate a hash for the data of the batch
                    seq_no = batch_dump["seq_no"]
                    data_hash = batch_dump["data_hash"]
                    batch_file = step_batches_dir / f"batch_{seq_no}_{data_hash}.json"

                    # Save the batch if it doesn't exist
                    if not batch_file.exists():
                        # Get the data of the batch before saving it
                        batch = next(
                            batch
                            for batch in self._steps[step_name].data[buffered_step_name]
                            if batch.seq_no == seq_no
                        )
                        batch_dump["data"] = batch.data
                        self.save(path=batch_file, format="json", dump=batch_dump)

                    keep_batches.append(batch_file)

                step_dump["data"][buffered_step_name] = [
                    str(file_batch) for file_batch in keep_batches
                ]

                # Remove `_Batch`es that were consumed from cache
                files = list_files_in_dir(step_batches_dir, key=None)
                remove = set(files) - set(keep_batches)
                for file in remove:
                    file.unlink()

            # Store the `_BatchManagerStep` info
            batch_manager_step_file = str(
                path.parent / f"batch_manager_steps/{step_name}/batch_manager_step.json"
            )
            self.save(path=batch_manager_step_file, format="json", dump=step_dump)

            # Store the path to the `_BatchManagerStep` file
            batch_manager_step_files[step_name] = batch_manager_step_file

        dump["steps"] = batch_manager_step_files
        self.save(path=path, format="json", dump=dump)

    @classmethod
    def load_from_cache(cls, path: "StrOrPath") -> "_BatchManager":
        """Loads the `_BatchManager` from a cache file.

        Args:
            path: The path to the cache file.
        """
        _check_is_dir(path)
        content = read_json(path)

        # Read each `_BatchManagerStep` from file
        steps = {}
        for step_name, step_file in content["steps"].items():
            steps[step_name] = read_json(step_file)

            # Read each `_Batch` from file
            for buffered_step_name, batch_files in steps[step_name]["data"].items():
                steps[step_name]["data"][buffered_step_name] = [
                    read_json(batch_file) for batch_file in batch_files
                ]

        content["steps"] = steps
        return cls.from_dict(content)


class _WriteBuffer:
    """Class in charge of sending the batched contents to a buffer and writing
    those to files under a given folder.

    As batches are received, they are added to the buffer and once each buffer
    is full, the content is written to a parquet file.
    """

    def __init__(self, path: "PathLike", leaf_steps: Set[str]) -> None:
        """
        Args:
            path: Folder where the files will be written, the idea
                is for this path to be in the cache folder under /data.
            leaf_steps: Leaf steps from either the DAG of the Pipeline.

        Raises:
            ValueError: If the path is not a directory.
        """
        self._path = Path(path)
        if not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)
            for step in leaf_steps:
                (self._path / step).mkdir(parents=True, exist_ok=True)

        if not self._path.is_dir():
            raise ValueError(f"The path should be a directory, not a file: {path}")

        self._buffers: Dict[str, List[Dict[str, Any]]] = {
            step: [] for step in leaf_steps
        }
        # TODO: make this configurable
        self._buffers_dump_batch_size: Dict[str, int] = {
            step: 50 for step in leaf_steps
        }
        self._buffer_last_schema = {}
        self._buffers_last_file: Dict[str, int] = {step: 1 for step in leaf_steps}
        self._logger = logging.getLogger("distilabel.write_buffer")

    def _get_filename(self, step_name: str) -> Path:
        """Creates the filename for the step.

        Args:
            step_name: Name of the step to which the data belongs to.

        Returns:
            Filename for the step.
        """
        return self._path / f"{step_name}.parquet"

    def is_full(self, step_name: str) -> bool:
        """Checks the buffers that are full so that those can be written to the file.

        Returns:
            Whether the buffer is full.
        """
        return len(self._buffers[step_name]) >= self._buffers_dump_batch_size[step_name]

    def add_batch(self, batch: "_Batch") -> None:
        """Adds a batch to the buffer and writes the buffer to the file if it's full.

        Args:
            batch: batch to add to the buffer.
        """
        step_name = batch.step_name
        data = batch.data[0]
        self._buffers[step_name].extend(data)
        self._logger.debug(
            f"Added batch to write buffer for step '{step_name}' with {len(data)} rows."
        )
        if self.is_full(step_name):
            self._logger.debug(
                f"Buffer for step '{step_name}' is full (rows: {len(self._buffers[step_name])},"
                f" full: {self._buffers_dump_batch_size[step_name]}), writing to file..."
            )
            self._write(step_name)

    def _write(self, step_name: str) -> None:
        """Writes the content to the file and cleans the buffer.

        Args:
            step_name (str): Name of the step to which the data pertains.
        """
        step_parquet_dir = Path(self._path, step_name)
        if not step_parquet_dir.exists():
            self._logger.debug(
                f"Creating directory for step '{step_name}' parquet files..."
            )
            step_parquet_dir.mkdir()

        table = pa.Table.from_pylist(self._buffers[step_name])

        last_schema = self._buffer_last_schema.get(step_name)
        if last_schema is None:
            self._buffer_last_schema[step_name] = table.schema
        else:
            if not last_schema.equals(table.schema):
                new_schema = pa.unify_schemas([last_schema, table.schema])
                self._buffer_last_schema[step_name] = new_schema
                table = table.cast(new_schema)

        next_file_number = self._buffers_last_file[step_name]
        self._buffers_last_file[step_name] = next_file_number + 1

        parquet_file = step_parquet_dir / f"{str(next_file_number).zfill(5)}.parquet"
        pq.write_table(table, parquet_file)
        self._logger.debug(f"Written to file '{parquet_file}'")

        self._clean_buffer(step_name)

    def _clean_buffer(self, step_name: str) -> None:
        """Cleans the buffer by setting it's content to `None`.

        Args:
            step_name: The name of the buffer to clean.
        """
        self._buffers[step_name] = []

    def close(self) -> None:
        """Closes the buffer by writing the remaining content to the file."""
        for step_name in self._buffers:
            if self._buffers[step_name]:
                self._write(step_name)

            # We need to read the parquet files and write them again to ensure the schema
            # is correct. Otherwise, the first parquets won't have the last schema and
            # then we will have issues when reading them.
            for file in list_files_in_dir(self._path / step_name):
                if step_name in self._buffer_last_schema:
                    table = pq.read_table(
                        file, schema=self._buffer_last_schema[step_name]
                    )
                    pq.write_table(table, file)
