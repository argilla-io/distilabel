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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
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

    from distilabel.distiset import Distiset
    from distilabel.pipeline.routing_batch_function import RoutingBatchFunction
    from distilabel.steps.base import _Step


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
    """

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

        self._fs: Optional[fsspec.AbstractFileSystem] = None
        self._storage_base_path: Optional[str] = None
        self._use_fs_to_pass_data: bool = False
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

    @abstractmethod
    def _send_to_step(self, step_name: str, to_send: Any) -> None:
        """Sends something to the input queue of a step.

        Args:
            step_name: The name of the step.
            to_send: The object to send.
        """
        pass

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

    def _notify_steps_to_stop(self) -> None:
        """Notifies the steps to stop their infinite running loop by sending `None` to
        their input queues."""
        for step_name in self.dag:
            self._send_to_step(step_name, None)


LAST_BATCH_SENT_FLAG = "last_batch_sent"
