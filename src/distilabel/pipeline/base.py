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
import logging
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import fsspec
from typing_extensions import Self
from upath import UPath

from distilabel import __version__
from distilabel.distiset import create_distiset
from distilabel.pipeline._dag import DAG
from distilabel.pipeline.batch import _Batch
from distilabel.pipeline.batch_manager import _BatchManager
from distilabel.pipeline.constants import (
    CONVERGENCE_STEP_ATTR_NAME,
    INPUT_QUEUE_ATTR_NAME,
    LAST_BATCH_SENT_FLAG,
    RECEIVES_ROUTED_BATCHES_ATTR_NAME,
    ROUTING_BATCH_FUNCTION_ATTR_NAME,
    STEP_ATTR_NAME,
)
from distilabel.pipeline.write_buffer import _WriteBuffer
from distilabel.utils.logging import setup_logging, stop_logging
from distilabel.utils.serialization import (
    TYPE_INFO_KEY,
    _Serializable,
)

if TYPE_CHECKING:
    from os import PathLike
    from queue import Queue

    from distilabel.distiset import Distiset
    from distilabel.pipeline.routing_batch_function import RoutingBatchFunction
    from distilabel.pipeline.typing import PipelineRuntimeParametersInfo, StepLoadStatus
    from distilabel.steps.base import Step, _Step


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
    batch_input_data: Path
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


_STEP_LOAD_FAILED_CODE = -666
_STEP_NOT_LOADED_CODE = -999


class BasePipeline(ABC, _Serializable):
    """Base class for a `distilabel` pipeline.

    Attributes:
        name: The name of the pipeline.
        description: A description of the pipeline.
        dag: The `DAG` instance that represents the pipeline.
        _cache_dir: The directory where the pipeline will be cached.
        _logger: The logger instance that will be used by the pipeline.
        _batch_manager: The batch manager that will manage the batches received from the
            steps while running the pipeline. It will be created when the pipeline is run,
            from scratch or from cache. Defaults to `None`.
        _write_buffer: The buffer that will store the data of the leaf steps of the pipeline
            while running, so the `Distiset` can be created at the end. It will be created
            when the pipeline is run. Defaults to `None`.
        _logging_parameters: A dictionary containing the parameters that will passed to
            `setup_logging` function to initialize the logging. Defaults to `{}`.
        _fs: The `fsspec` filesystem to be used to store the data of the `_Batch`es passed
            between the steps. It will be set when the pipeline is run. Defaults to `None`.
        _storage_base_path: The base path where the data of the `_Batch`es passed between
            the steps will be stored. It will be set then the pipeline is run. Defaults
            to `None`.
        _use_fs_to_pass_data: Whether to use the file system to pass the data of the
            `_Batch`es between the steps. Even if this parameter is `False`, the `Batch`es
            received by `GlobalStep`s will always use the file system to pass the data.
            Defaults to `False`.
        _dry_run: A flag to indicate if the pipeline is running in dry run mode. Defaults
            to `False`.
        output_queue: A queue to store the output of the steps while running the pipeline.
        load_queue: A queue used by each `Step` to notify the main process it has finished
            loading or it the step has been unloaded.
    """

    _output_queue: "Queue[Any]"
    _load_queue: "Queue[Union[StepLoadStatus, None]]"

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        cache_dir: Optional[Union[str, "PathLike"]] = None,
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

        self._batch_manager: Optional["_BatchManager"] = None
        self._write_buffer: Optional["_WriteBuffer"] = None
        self._logging_parameters: Dict[str, Any] = {
            "filename": self._cache_location["log_file"]
        }

        self._steps_load_status: Dict[str, int] = {}
        self._steps_load_status_lock = threading.Lock()

        self._stop_called = False
        self._stop_called_lock = threading.Lock()
        self._stop_calls = 0

        self._fs: Optional[fsspec.AbstractFileSystem] = None
        self._storage_base_path: Optional[str] = None
        self._use_fs_to_pass_data: bool = False

        self._dry_run = False

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

    def _set_logging_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set the parameters that will be passed to the `setup_logging` function to
        initialize the logging.

        Args:
            parameters: A dictionary with the parameters that will be passed to the
                `setup_logging` function.
        """
        self._logging_parameters = parameters

    def run(
        self,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache: bool = True,
        storage_parameters: Optional[Dict[str, Any]] = None,
        use_fs_to_pass_data: bool = False,
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
            storage_parameters: A dictionary with the storage parameters (`fsspec` and path)
                that will be used to store the data of the `_Batch`es passed between the
                steps if `use_fs_to_pass_data` is `True` (for the batches received by a
                `GlobalStep` it will be always used). It must have at least the "path" key,
                and it can contain additional keys depending on the protocol. By default,
                it will use the local file system and a directory in the cache directory.
                Defaults to `None`.
            use_fs_to_pass_data: Whether to use the file system to pass the data of
                the `_Batch`es between the steps. Even if this parameter is `False`, the
                `Batch`es received by `GlobalStep`s will always use the file system to
                pass the data. Defaults to `False`.

        Returns:
            The `Distiset` created by the pipeline.
        """

        # Set the runtime parameters that will be used during the pipeline execution.
        # They are used to generate the signature of the pipeline that is used to hit the
        # cache when the pipeline is run, so it's important to do it first.
        self._set_runtime_parameters(parameters or {})

        setup_logging(
            **{
                **self._logging_parameters,
                "filename": str(self._cache_location["log_file"]),
            }
        )

        self._init_steps_load_status()

        # Validate the pipeline DAG to check that all the steps are chainable, there are
        # no missing runtime parameters, batch sizes are correct, etc.
        self.dag.validate()

        # Load the `_BatchManager` from cache or create one from scratch
        self._load_batch_manager(use_cache)

        # Setup the filesystem that will be used to pass the data of the `_Batch`es
        self._setup_fsspec(storage_parameters)
        self._use_fs_to_pass_data = use_fs_to_pass_data

        if self._dry_run:
            self._logger.info("🌵 Dry run mode")

        # If the batch manager is not able to generate batches, that means that the loaded
        # `_BatchManager` from cache didn't have any remaining batches to process i.e.
        # the previous pipeline execution was completed successfully.
        if not self._batch_manager.can_generate():  # type: ignore
            self._logger.info(
                "💾 Loaded batch manager from cache doesn't contain any remaining data."
                " Returning `Distiset` from cache data..."
            )
            stop_logging()
            return create_distiset(
                self._cache_location["data"],
                pipeline_path=self._cache_location["pipeline"],
                log_filename_path=self._cache_location["log_file"],
                enable_metadata=self._enable_metadata,
            )

        self._setup_write_buffer()

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

    def get_runtime_parameters_info(self) -> "PipelineRuntimeParametersInfo":
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

    def _init_steps_load_status(self) -> None:
        """Initialize the `_steps_load_status` dictionary assigning 0 to every step of
        the pipeline."""
        for step_name in self.dag:
            self._steps_load_status[step_name] = _STEP_NOT_LOADED_CODE

    def _setup_fsspec(
        self, storage_parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Setups the `fsspec` filesystem to be used to store the data of the `_Batch`es
        passed between the steps.

        Args:
            storage_parameters: A dictionary with the storage parameters (`fsspec` and path)
                that will be used to store the data of the `_Batch`es passed between the
                steps if `use_fs_to_pass_data` is `True` (for the batches received by a
                `GlobalStep` it will be always used). It must have at least the "path" key,
                and it can contain additional keys depending on the protocol. By default,
                it will use the local file system and a directory in the cache directory.
                Defaults to `None`.
        """
        if not storage_parameters:
            self._fs = fsspec.filesystem("file")
            self._storage_base_path = (
                f"file://{self._cache_location['batch_input_data']}"
            )
            return

        if "path" not in storage_parameters:
            raise ValueError(
                "The 'path' key must be present in the `storage_parameters` dictionary"
                " if it's not `None`."
            )

        path = storage_parameters.pop("path")
        protocol = UPath(path).protocol

        self._fs = fsspec.filesystem(protocol, **storage_parameters)
        self._storage_base_path = path

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

    def _is_convergence_step(self, step_name: str) -> None:
        """Checks if a step is a convergence step.

        Args:
            step_name: The name of the step.
        """
        return self.dag.get_step(step_name).get(CONVERGENCE_STEP_ATTR_NAME)

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
            "batch_input_data": folder / "batch_input_data",
            "log_file": folder / "pipeline.log",
        }

    def _cache(self) -> None:
        """Saves the `BasePipeline` using the `_cache_filename`."""
        if self._dry_run:
            return

        self.save(
            path=self._cache_location["pipeline"],
            format=self._cache_location["pipeline"].suffix.replace(".", ""),  # type: ignore
        )
        if self._batch_manager is not None:
            self._batch_manager.cache(self._cache_location["batch_manager"])
        self._logger.debug("Pipeline and batch manager saved to cache.")

    def _load_batch_manager(self, use_cache: bool = True) -> None:
        """Will try to load the `_BatchManager` from the cache dir if found. Otherwise,
        it will create one from scratch.
        """
        batch_manager_cache_loc = self._cache_location["batch_manager"]
        if use_cache and batch_manager_cache_loc.exists():
            self._logger.info(
                f"💾 Loading `_BatchManager` from cache: '{batch_manager_cache_loc}'"
            )
            self._batch_manager = _BatchManager.load_from_cache(batch_manager_cache_loc)
        else:
            self._batch_manager = _BatchManager.from_dag(self.dag)

    def _setup_write_buffer(self) -> None:
        """Setups the `_WriteBuffer` that will store the data of the leaf steps of the
        pipeline while running, so the `Distiset` can be created at the end.
        """
        buffer_data_path = self._cache_location["data"]
        self._logger.info(f"📝 Pipeline data will be written to '{buffer_data_path}'")
        self._write_buffer = _WriteBuffer(buffer_data_path, self.dag.leaf_steps)

    def _run_output_queue_loop_in_thread(self) -> threading.Thread:
        """Runs the output queue loop in a separate thread to receive the output batches
        from the steps. This is done to avoid the signal handler to block the loop, which
        would prevent the pipeline from stopping correctly."""
        thread = threading.Thread(target=self._output_queue_loop)
        thread.start()
        return thread

    def _output_queue_loop(self) -> None:
        """Loop to receive the output batches from the steps and manage the flow of the
        batches through the pipeline."""
        while self._batch_manager.can_generate() and not self._stop_called:  # type: ignore
            self._logger.debug("Waiting for output batch from step...")
            if (batch := self._output_queue.get()) is None:
                self._logger.debug("Received `None` from output queue. Breaking loop.")
                break

            self._logger.debug(
                f"Received batch with seq_no {batch.seq_no} from step '{batch.step_name}'"
                f" from output queue: {batch}"
            )

            if batch.data_path:
                self._logger.debug(
                    f"Reading {batch.seq_no} batch data from '{batch.step_name}': '{batch.data_path}'"
                )
                batch.read_batch_data_from_fs()

            if batch.step_name in self.dag.leaf_steps:
                self._write_buffer.add_batch(batch)  # type: ignore

            # If `_stop_called` was set to `True` while waiting for the output queue, then
            # we need to handle the stop of the pipeline and break the loop to avoid
            # propagating the batches through the pipeline and making the stop process
            # slower.
            if self._stop_called:
                self._handle_batch_on_stop(batch)
                break

            self._manage_batch_flow(batch)

        if self._stop_called:
            self._handle_stop()

        self._cache()

    def _run_load_queue_loop_in_thread(self) -> threading.Thread:
        """Runs a background thread that reads from the `load_queue` to update the status
        of the number of workers loaded for each step.

        Returns:
            The thread that was started.
        """
        thread = threading.Thread(target=self._run_load_queue_loop)
        thread.start()
        return thread

    def _run_load_queue_loop(self) -> None:
        """Runs a loop that reads from the `load_queue` to update the status of the number
        of workers loaded for each step."""
        while True:
            if (load_info := self._load_queue.get()) is None:
                self._logger.debug("Received `None` from load queue. Breaking loop.")
                break

            with self._steps_load_status_lock:
                step_name, status = load_info["name"], load_info["status"]
                if status == "loaded":
                    if self._steps_load_status[step_name] == _STEP_NOT_LOADED_CODE:
                        self._steps_load_status[step_name] = 1
                    else:
                        self._steps_load_status[step_name] += 1
                elif status == "unloaded":
                    self._steps_load_status[step_name] -= 1
                else:
                    # load failed
                    self._steps_load_status[step_name] = _STEP_LOAD_FAILED_CODE

                self._logger.debug(
                    f"Step '{step_name}' loaded workers: {self._steps_load_status[step_name]}"
                )

    def _all_steps_loaded(self) -> bool:
        """Waits for all the steps to load.

        Returns:
            `True` if all the steps have been loaded correctly, `False` otherwise.
        """

        self._logger.info("⏳ Waiting for all the steps to load...")
        previous_message = None
        while not self._stop_called:
            with self._steps_load_status_lock:
                self._logger.debug(f"Steps loaded: {self._steps_load_status}")

                if any(
                    num_workers_loaded == _STEP_LOAD_FAILED_CODE
                    for num_workers_loaded in self._steps_load_status.values()
                ):
                    self._logger.error("❌ Failed to load all the steps")
                    return False

                num_steps_loaded = 0
                workers_message = ""
                for step_name, num_workers_loaded in self._steps_load_status.items():
                    # TODO: update condition once we allow more than one worker per step
                    if num_workers_loaded == 1:
                        num_steps_loaded += 1
                    workers_message += (
                        f"\n * '{step_name}' workers: {max(0, num_workers_loaded)}"
                    )

                message = f"⏳ Steps loaded: {num_steps_loaded}/{len(self.dag)}{workers_message}"
                if num_steps_loaded > 0 and message != previous_message:
                    self._logger.info(message)
                    previous_message = message

                if num_steps_loaded == len(self.dag):
                    self._logger.info("✅ All the steps have been loaded!")
                    return True

            time.sleep(2.5)

        return not self._stop_called

    def _handle_stop(self) -> None:
        """Handles the stop of the pipeline execution, which will stop the steps from
        processing more batches and wait for the output queue to be empty, to not lose
        any data that was already processed by the steps before the stop was called."""
        self._logger.debug("Handling stop of the pipeline execution...")

        self._add_batches_back_to_batch_manager()

        # Wait for the input queue to be empty, which means that all the steps finished
        # processing the batches that were sent before the stop flag.
        for step_name in self.dag:
            self._wait_step_input_queue_empty(step_name)

        self._consume_output_queue()

    def _wait_step_input_queue_empty(self, step_name: str) -> Union["Queue[Any]", None]:
        """Waits for the input queue of a step to be empty.

        Args:
            step_name: The name of the step.

        Returns:
            The input queue of the step if it's not loaded or finished, `None` otherwise.
        """
        if self._check_step_not_loaded_or_finished(step_name):
            return None

        if input_queue := self.dag.get_step(step_name).get(INPUT_QUEUE_ATTR_NAME):
            while input_queue.qsize() != 0:
                pass
            return input_queue

    def _check_step_not_loaded_or_finished(self, step_name: str) -> bool:
        """Checks if a step is not loaded or already finished.

        Args:
            step_name: The name of the step.

        Returns:
            `True` if the step is not loaded or already finished, `False` otherwise.
        """
        with self._steps_load_status_lock:
            num_workers = self._steps_load_status[step_name]

            # The step has finished (workers = 0) or it has failed to load
            if num_workers in [0, _STEP_LOAD_FAILED_CODE]:
                return True

        return False

    @property
    @abstractmethod
    def QueueClass(self) -> Callable:
        """The class of the queue to use in the pipeline."""
        pass

    def _create_step_input_queue(self, step_name: str) -> "Queue[Any]":
        """Creates an input queue for a step.

        Args:
            step_name: The name of the step.

        Returns:
            The input queue created.
        """
        input_queue = self.QueueClass()
        self.dag.set_step_attr(step_name, INPUT_QUEUE_ATTR_NAME, input_queue)
        return input_queue

    @abstractmethod
    def _run_step(self, step: "_Step", input_queue: "Queue[Any]") -> None:
        """Runs the `Step` instance.

        Args:
            step: The `Step` instance to run.
            input_queue: The input queue where the step will receive the batches.
        """
        pass

    def _run_steps(self) -> None:
        """Runs the `Step`s of the pipeline, creating first an input queue for each step
        that will be used to send the batches.
        """
        for step_name in self.dag:
            step: "Step" = self.dag.get_step(step_name)[STEP_ATTR_NAME]
            input_queue = self._create_step_input_queue(step_name=step_name)

            # Set `pipeline` to `None` as in some Python environments the pipeline is not
            # picklable and it will raise an error when trying to send the step to the process.
            # `TypeError: cannot pickle 'code' object`
            step.pipeline = None

            self._logger.debug(f"Running 1 instance of step '{step.name}'...")
            self._run_step(step=step, input_queue=input_queue)

    def _add_batches_back_to_batch_manager(self) -> None:
        """Add the `Batch`es that were sent to a `Step` back to the `_BatchManager`. This
        method should be used when the pipeline has been stopped prematurely."""
        for step_name in self.dag:
            node = self.dag.get_step(step_name)
            step: "_Step" = node[STEP_ATTR_NAME]
            if step.is_generator:
                continue
            if input_queue := node.get(INPUT_QUEUE_ATTR_NAME):
                while not input_queue.empty():
                    batch = input_queue.get()
                    if batch is None:
                        continue
                    self._batch_manager.add_batch(  # type: ignore
                        to_step=step_name, batch=batch, prepend=True
                    )
                    self._logger.debug(
                        f"Adding batch back to the batch manager: {batch}"
                    )
                input_queue.put(None)

    def _consume_output_queue(self) -> None:
        """Consumes the `Batch`es from the output queue until it's empty. This method should
        be used when the pipeline has been stopped prematurely to consume and to not lose
        the `Batch`es that were processed by the leaf `Step`s before stopping the pipeline."""
        while not self._output_queue.empty():
            batch = self._output_queue.get()
            if batch is None:
                continue

            if batch.step_name in self.dag.leaf_steps:
                self._write_buffer.add_batch(batch)  # type: ignore

            self._handle_batch_on_stop(batch)

    def _manage_batch_flow(self, batch: "_Batch") -> None:
        """Checks if the step that generated the batch has more data in its buffer to
        generate a new batch. If there's data, then a new batch is sent to the step. If
        the step has no data in its buffer, then the predecessors generator steps are
        requested to send a new batch.

        Args:
            batch: The batch that was processed.
        """
        assert self._batch_manager, "Batch manager is not set"

        # Make sure to send the `LAST_BATCH_SENT_FLAG` to the predecessors of the convergence
        # step if the batch is the last one, so they stop their processing loop even if
        # they haven't received the last batch because of the routing function.
        if self._is_convergence_step(batch.step_name) and batch.last_batch:
            for step_name in self.dag.get_step_predecessors(batch.step_name):
                self._send_last_batch_flag_to_step(step_name)

        route_to, routed = self._get_successors(batch)

        # Keep track of the steps that the batch was routed to
        if routed:
            batch.batch_routed_to = route_to

        self._register_batch(batch)

        step = self._get_step_from_batch(batch)

        # Add the batch to the successors input buffers
        for successor in route_to:
            # Copy batch to avoid modifying the same reference in the batch manager
            batch_to_add = batch.copy() if len(route_to) > 1 else batch

            self._batch_manager.add_batch(successor, batch_to_add)

            # Check if the step is a generator and if there are successors that need data
            # from this step. This usually happens when the generator `batch_size` is smaller
            # than the `input_batch_size` of the successor steps.
            if (
                step.is_generator
                and step.name in self._batch_manager.step_empty_buffers(successor)
            ):
                last_batch_sent = self._batch_manager.get_last_batch_sent(step.name)
                self._send_batch_to_step(last_batch_sent.next_batch())  # type: ignore

            # If successor step has enough data in its buffer to create a new batch, then
            # send the batch to the step.
            if new_batch := self._batch_manager.get_batch(successor):
                self._send_batch_to_step(new_batch)

        if not step.is_generator:
            # Step ("this", the one from which the batch was received) has enough data on its
            # buffers to create a new batch
            if new_batch := self._batch_manager.get_batch(step.name):  # type: ignore
                self._send_batch_to_step(new_batch)
            else:
                self._request_more_batches_if_needed(step)

        self._cache()

    def _send_to_step(self, step_name: str, to_send: Any) -> None:
        """Sends something to the input queue of a step.

        Args:
            step_name: The name of the step.
            to_send: The object to send.
        """
        input_queue = self.dag.get_step(step_name)[INPUT_QUEUE_ATTR_NAME]
        input_queue.put(to_send)

    def _send_batch_to_step(self, batch: "_Batch") -> None:
        """Sends a batch to the input queue of a step, writing the data of the batch
        to the filesystem and setting `batch.data_path` with the path where the data
        was written (if requiered i.e. the step is a global step or `use_fs_to_pass_data`)

        This method should be extended by the specific pipeline implementation, adding
        the logic to send the batch to the step.

        Args:
            batch: The batch to send.
        """
        self._logger.debug(
            f"Setting batch {batch.seq_no} as last batch sent to '{batch.step_name}': {batch}"
        )
        self._batch_manager.set_last_batch_sent(batch)  # type: ignore

        step: "_Step" = self.dag.get_step(batch.step_name)[STEP_ATTR_NAME]
        if not step.is_generator and (step.is_global or self._use_fs_to_pass_data):
            base_path = UPath(self._storage_base_path) / step.name  # type: ignore
            self._logger.debug(
                f"Writing {batch.seq_no} batch for '{batch.step_name}' step to filesystem: {base_path}"
            )
            batch.write_batch_data_to_fs(self._fs, base_path)  # type: ignore

        self._logger.debug(
            f"Sending batch {batch.seq_no} to step '{batch.step_name}': {batch}"
        )

        self._send_to_step(batch.step_name, batch)

    def _register_batch(self, batch: "_Batch") -> None:
        """Registers a batch in the batch manager.

        Args:
            batch: The batch to register.
        """
        self._batch_manager.register_batch(batch)  # type: ignore
        self._logger.debug(
            f"Batch {batch.seq_no} from step '{batch.step_name}' registered in batch"
            " manager"
        )

    def _send_last_batch_flag_to_step(self, step_name: str) -> None:
        """Sends the `LAST_BATCH_SENT_FLAG` to a step to stop processing batches.

        Args:
            step_name: The name of the step.
        """
        batch = self._batch_manager.get_last_batch_sent(step_name)  # type: ignore
        if batch and batch.last_batch:
            return

        self._logger.debug(
            f"Sending `LAST_BATCH_SENT_FLAG` to '{step_name}' step to stop processing"
            " batches..."
        )

        self._send_to_step(step_name, LAST_BATCH_SENT_FLAG)
        self._batch_manager.set_last_batch_flag_sent_to(step_name)  # type: ignore

    def _request_initial_batches(self) -> None:
        """Requests the initial batches to the generator steps."""
        assert self._batch_manager, "Batch manager is not set"

        for step in self._batch_manager._steps.values():
            if batch := step.get_batch():
                self._logger.debug(
                    f"Sending initial batch to '{step.step_name}' step: {batch}"
                )
                self._send_batch_to_step(batch)

        for step_name in self.dag.root_steps:
            seq_no = 0
            if last_batch := self._batch_manager.get_last_batch(step_name):
                seq_no = last_batch.seq_no + 1
            batch = _Batch(seq_no=seq_no, step_name=step_name, last_batch=self._dry_run)
            self._logger.debug(
                f"Requesting initial batch to '{step_name}' generator step: {batch}"
            )
            self._send_batch_to_step(batch)

    def _request_more_batches_if_needed(self, step: "Step") -> None:
        """Request more batches to the predecessors steps of `step` if needed.

        Args:
            step: The step of which it has to be checked if more batches are needed from
                its predecessors.
        """
        empty_buffers = self._batch_manager.step_empty_buffers(step.name)  # type: ignore
        for previous_step_name in empty_buffers:
            # Only more batches can be requested to the `GeneratorStep`s as they are the
            # only kind of steps that lazily generate batches.
            if previous_step_name not in self.dag.root_steps:
                continue

            # Get the last batch that the previous step sent to generate the next batch
            # (next `seq_no`).
            last_batch = self._batch_manager.get_last_batch_sent(previous_step_name)  # type: ignore
            if last_batch is None:
                continue

            self._logger.debug(
                f"Step '{step.name}' input buffer for step '{previous_step_name}' is"
                " empty. Requesting new batch..."
            )
            self._send_batch_to_step(last_batch.next_batch())

    def _handle_batch_on_stop(self, batch: "_Batch") -> None:
        """Handles a batch that was received from the output queue when the pipeline was
        stopped. It will add and register the batch in the batch manager.

        Args:
            batch: The batch to handle.
        """
        assert self._batch_manager, "Batch manager is not set"

        self._batch_manager.register_batch(batch)
        step: "Step" = self.dag.get_step(batch.step_name)[STEP_ATTR_NAME]
        for successor in self.dag.get_step_successors(step.name):  # type: ignore
            self._batch_manager.add_batch(successor, batch)

    def _get_step_from_batch(self, batch: "_Batch") -> "Step":
        """Gets the `Step` instance from a batch.

        Args:
            batch: The batch to get the step from.

        Returns:
            The `Step` instance.
        """
        return self.dag.get_step(batch.step_name)[STEP_ATTR_NAME]

    def _notify_steps_to_stop(self) -> None:
        """Notifies the steps to stop their infinite running loop by sending `None` to
        their input queues."""
        for step_name in self.dag:
            self._send_to_step(step_name, None)

    def _get_successors(self, batch: "_Batch") -> Tuple[List[str], bool]:
        """Gets the successors and the successors to which the batch has to be routed.

        Args:
            batch: The batch to which the successors will be determined.

        Returns:
            The successors to route the batch to and whether the batch was routed using
            a routing function.
        """
        node = self.dag.get_step(batch.step_name)
        step: "Step" = node[STEP_ATTR_NAME]
        successors = list(self.dag.get_step_successors(step.name))  # type: ignore
        route_to = successors

        # Check if the step has a routing function to send the batch to specific steps
        if routing_batch_function := node.get(ROUTING_BATCH_FUNCTION_ATTR_NAME):
            route_to = routing_batch_function(batch, successors)
            successors_str = ", ".join(f"'{successor}'" for successor in route_to)
            self._logger.info(
                f"🚏 Using '{step.name}' routing function to send batch {batch.seq_no} to steps: {successors_str}"
            )

        return route_to, route_to != successors

    @abstractmethod
    def _stop(self) -> None:
        """Stops the pipeline in a controlled way."""
        pass

    def _stop_load_queue_loop(self) -> None:
        """Stops the `_load_queue` loop sending a `None`."""
        self._logger.debug("Sending `None` to the load queue to notify stop...")
        self._load_queue.put(None)

    def _stop_output_queue_loop(self) -> None:
        """Stops the `_output_queue` loop sending a `None`."""
        self._logger.debug("Sending `None` to the output queue to notify stop...")
        self._output_queue.put(None)

    def _handle_keyboard_interrupt(self) -> Any:
        """Handles KeyboardInterrupt signal sent during the Pipeline.run method.

        It will try to call self._stop (if the pipeline didn't started yet, it won't
        have any effect), and if the pool is already started, will close it before exiting
        the program.

        Returns:
            The original `signal.SIGINT` handler.
        """

        def signal_handler(signumber: int, frame: Any) -> None:
            self._stop()

        return signal.signal(signal.SIGINT, signal_handler)
