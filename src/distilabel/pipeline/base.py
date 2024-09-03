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
from inspect import isclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import fsspec
from pydantic import BaseModel
from typing_extensions import Self
from upath import UPath

from distilabel import __version__, constants, envs
from distilabel.distiset import create_distiset
from distilabel.errors import DistilabelUserError
from distilabel.mixins.requirements import RequirementsMixin
from distilabel.pipeline._dag import DAG
from distilabel.pipeline.batch import _Batch
from distilabel.pipeline.batch_manager import _BatchManager
from distilabel.pipeline.write_buffer import _WriteBuffer
from distilabel.steps.base import GeneratorStep
from distilabel.steps.generators.utils import make_generator_step
from distilabel.utils.logging import setup_logging, stop_logging
from distilabel.utils.serialization import (
    TYPE_INFO_KEY,
    _Serializable,
    read_json,
)
from distilabel.utils.typing_ import (
    extract_annotation_inner_type,
    is_type_pydantic_secret_field,
)

if TYPE_CHECKING:
    from os import PathLike
    from queue import Queue

    from pydantic import BaseModel

    from distilabel.distiset import Distiset
    from distilabel.pipeline.routing_batch_function import RoutingBatchFunction
    from distilabel.pipeline.typing import (
        InputDataset,
        PipelineRuntimeParametersInfo,
        StepLoadStatus,
    )
    from distilabel.steps.base import Step, _Step

    class _CacheLocation(TypedDict):
        """Dictionary to store the filenames and directories of a cached pipeline.

        Attributes:
            base: The base directory where all the files of the pipeline will be stored.
            pipeline: The filename where the pipeline content will be serialized.
            batch_manager: The filename where the batch manager content will be serialized.
            data: The directory where the output data of each leaf step will be stored.
            batch_input_data: The directory where the input data of the steps using the
                file system to pass the data will be stored.
            log_file: The filename where the logs will be stored.
            stages_file: The filename where the stages status will be stored.
        """

        base: Path
        pipeline: Path
        batch_manager: Path
        data: Path
        batch_input_data: Path
        log_file: Path
        stages_file: Path


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

_ATTRIBUTES_IGNORED_CACHE = ("disable_cuda_device_placement", "jobs_ids")
_PIPELINE_DEFAULT_NAME = "__default_pipeline_name__"


class BasePipeline(ABC, RequirementsMixin, _Serializable):
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
        name: Optional[str] = None,
        description: Optional[str] = None,
        cache_dir: Optional[Union[str, "PathLike"]] = None,
        enable_metadata: bool = False,
        requirements: Optional[List[str]] = None,
    ) -> None:
        """Initialize the `BasePipeline` instance.

        Args:
            name: The name of the pipeline. If not generated, a random one will be generated by default.
            description: A description of the pipeline. Defaults to `None`.
            cache_dir: A directory where the pipeline will be cached. Defaults to `None`.
            enable_metadata: Whether to include the distilabel metadata column for the pipeline
                in the final `Distiset`. It contains metadata used by distilabel, for example
                the raw outputs of the `LLM` without processing would be here, inside `raw_output_...`
                field. Defaults to `False`.
            requirements: List of requirements that must be installed to run the pipeline.
                Defaults to `None`, but can be helpful to inform in a pipeline to be shared
                that this requirements must be installed.
        """
        self.name = name or _PIPELINE_DEFAULT_NAME
        self.description = description
        self._enable_metadata = enable_metadata
        self.dag = DAG()

        if cache_dir:
            self._cache_dir = Path(cache_dir)
        elif env_cache_dir := envs.DISTILABEL_CACHE_DIR:
            self._cache_dir = Path(env_cache_dir)
        else:
            self._cache_dir = constants.PIPELINE_CACHE_DIR

        self._logger = logging.getLogger("distilabel.pipeline")

        self._batch_manager: Optional["_BatchManager"] = None
        self._write_buffer: Optional["_WriteBuffer"] = None

        self._steps_load_status: Dict[str, int] = {}
        self._steps_load_status_lock = threading.Lock()

        self._stop_called = False
        self._stop_called_lock = threading.Lock()
        self._stop_calls = 0

        self._recover_offline_batch_generate_for_step: Union[
            Tuple[str, List[List[Dict[str, Any]]]], None
        ] = None

        self._fs: Optional[fsspec.AbstractFileSystem] = None
        self._storage_base_path: Optional[str] = None
        self._use_fs_to_pass_data: bool = False
        self._dry_run = False

        self._current_stage = 0
        self._stages_last_batch: List[List[str]] = []

        self.requirements = requirements or []

        self._exception: Union[Exception, None] = None

        self._log_queue: Union["Queue[Any]", None] = None

    def __enter__(self) -> Self:
        """Set the global pipeline instance when entering a pipeline context."""
        _GlobalPipelineManager.set_pipeline(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Unset the global pipeline instance when exiting a pipeline context."""
        _GlobalPipelineManager.set_pipeline(None)
        self._set_pipeline_name()

    def _set_pipeline_name(self) -> None:
        """Creates a name for the pipeline if it's the default one (if hasn't been set)."""
        if self.name == _PIPELINE_DEFAULT_NAME:
            self.name = f"pipeline_{'_'.join(self.dag)}"

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
            for argument, value in sorted(step[constants.STEP_ATTR_NAME].items()):
                if (argument == TYPE_INFO_KEY) or (value is None):
                    continue

                if isinstance(value, dict):
                    # input_mappings/output_mappings
                    step_info += "-".join(
                        [
                            f"{str(k)}={str(v)}"
                            for k, v in value.items()
                            if k not in _ATTRIBUTES_IGNORED_CACHE
                        ]
                    )
                elif isinstance(value, (list, tuple)):
                    # runtime_parameters_info
                    step_info += "-".join([str(v) for v in value])
                elif isinstance(value, (int, str, float, bool)):
                    if argument not in _ATTRIBUTES_IGNORED_CACHE:
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
                constants.ROUTING_BATCH_FUNCTION_ATTR_NAME
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
        storage_parameters: Optional[Dict[str, Any]] = None,
        use_fs_to_pass_data: bool = False,
        dataset: Optional["InputDataset"] = None,
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
            dataset: If given, it will be used to create a `GeneratorStep` and put it as the
                root step. Convenient method when you have already processed the dataset in
                your script and just want to pass it already processed. Defaults to `None`.

        Returns:
            The `Distiset` created by the pipeline.
        """

        self._exception: Union[Exception, None] = None

        # Set the runtime parameters that will be used during the pipeline execution.
        # They are used to generate the signature of the pipeline that is used to hit the
        # cache when the pipeline is run, so it's important to do it first.
        self._set_runtime_parameters(parameters or {})

        self._refresh_pipeline_from_cache()

        if dataset is not None:
            self._add_dataset_generator_step(dataset)

        setup_logging(
            log_queue=self._log_queue, filename=str(self._cache_location["log_file"])
        )

        # Set the name of the pipeline if it's the default one. This should be called
        # if the pipeline is defined within the context manager, and the run is called
        # outside of it. Is here in the following case:
        # with Pipeline() as pipeline:
        #    pipeline.run()
        self._set_pipeline_name()

        # Validate the pipeline DAG to check that all the steps are chainable, there are
        # no missing runtime parameters, batch sizes are correct, etc.
        self.dag.validate()

        self._set_pipeline_artifacts_path_in_steps()

        # Set the initial load status for all the steps
        self._init_steps_load_status()

        # Load the stages status or initialize it
        self._load_stages_status(use_cache)

        # Load the `_BatchManager` from cache or create one from scratch
        self._load_batch_manager(use_cache)

        # Check pipeline requirements are installed
        self._check_requirements()

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
            distiset = create_distiset(
                data_dir=self._cache_location["data"],
                pipeline_path=self._cache_location["pipeline"],
                log_filename_path=self._cache_location["log_file"],
                enable_metadata=self._enable_metadata,
                dag=self.dag,
            )
            stop_logging()
            return distiset

        self._setup_write_buffer()

        self._print_load_stages_info()

    def dry_run(
        self,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 1,
        dataset: Optional["InputDataset"] = None,
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
            dataset: If given, it will be used to create a `GeneratorStep` and put it as the
                root step. Convenient method when you have already processed the dataset in
                your script and just want to pass it already processed. Defaults to `None`.

        Returns:
            Will return the `Distiset` as the main run method would do.
        """
        self._dry_run = True

        for step_name in self.dag:
            step = self.dag.get_step(step_name)[constants.STEP_ATTR_NAME]

            if step.is_generator:
                if not parameters:
                    parameters = {}
                parameters[step_name] = {"batch_size": batch_size}

        distiset = self.run(parameters=parameters, use_cache=False, dataset=dataset)

        self._dry_run = False
        return distiset

    def _add_dataset_generator_step(self, dataset: "InputDataset") -> None:
        """Create a root step to work as the `GeneratorStep` for the pipeline using a
        dataset.

        Args:
            dataset: A dataset that will be used to create a `GeneratorStep` and
                placed in the DAG as the root step.

        Raises:
            ValueError: If there's already a `GeneratorStep` in the pipeline.
        """
        for step_name in self.dag:
            step = self.dag.get_step(step_name)[constants.STEP_ATTR_NAME]
            if isinstance(step_name, GeneratorStep):
                raise DistilabelUserError(
                    "There is already a `GeneratorStep` in the pipeline, you can either"
                    " pass a `dataset` to the run method, or create a `GeneratorStep` explictly."
                    f" `GeneratorStep`: {step}",
                    page="sections/how_to_guides/basic/step/#types-of-steps",
                )
        loader = make_generator_step(dataset, self)
        self.dag.add_root_step(loader)

    def get_runtime_parameters_info(self) -> "PipelineRuntimeParametersInfo":
        """Get the runtime parameters for the steps in the pipeline.

        Returns:
            A dictionary with the step name as the key and a list of dictionaries with
            the parameter name and the parameter info as the value.
        """
        runtime_parameters = {}
        for step_name in self.dag:
            step: "_Step" = self.dag.get_step(step_name)[constants.STEP_ATTR_NAME]
            runtime_parameters[step_name] = step.get_runtime_parameters_info()
        return runtime_parameters

    def _init_steps_load_status(self) -> None:
        """Initialize the `_steps_load_status` dictionary assigning 0 to every step of
        the pipeline."""
        for step_name in self.dag:
            self._steps_load_status[step_name] = _STEP_NOT_LOADED_CODE

    def _set_pipeline_artifacts_path_in_steps(self) -> None:
        """Sets the attribute `_pipeline_artifacts_path` in all the `Step`s of the pipeline,
        so steps can use it to get the path to save the generated artifacts."""
        artifacts_path = self._cache_location["data"] / constants.STEPS_ARTIFACTS_PATH
        for name in self.dag:
            step: "_Step" = self.dag.get_step(name)[constants.STEP_ATTR_NAME]
            step.set_pipeline_artifacts_path(path=artifacts_path)

    def _check_requirements(self) -> None:
        """Checks if the dependencies required to run the pipeline are installed.

        Raises:
            ModuleNotFoundError: if one or more requirements are missing.
        """
        if to_install := self.requirements_to_install():
            # Print the list of requirements like they would appear in a requirements.txt
            to_install_list = "\n" + "\n".join(to_install)
            msg = f"Please install the following requirements to run the pipeline: {to_install_list}"
            self._logger.error(msg)
            raise ModuleNotFoundError(msg)

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
            raise DistilabelUserError(
                "The 'path' key must be present in the `storage_parameters` dictionary"
                " if it's not `None`.",
                page="sections/how_to_guides/advanced/fs_to_pass_data/",
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
        routing_batch_function = node.get(
            constants.ROUTING_BATCH_FUNCTION_ATTR_NAME, None
        )
        self.dag.set_step_attr(
            name=to_step,
            attr=constants.RECEIVES_ROUTED_BATCHES_ATTR_NAME,
            value=routing_batch_function is not None,
        )

    def _is_convergence_step(self, step_name: str) -> None:
        """Checks if a step is a convergence step.

        Args:
            step_name: The name of the step.
        """
        return self.dag.get_step(step_name).get(constants.CONVERGENCE_STEP_ATTR_NAME)

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
            attr=constants.ROUTING_BATCH_FUNCTION_ATTR_NAME,
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
                step: "_Step" = self.dag.get_step(step_name)[constants.STEP_ATTR_NAME]
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
            "requirements": self.requirements,
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
        requirements = data.get("requirements", [])
        with cls(name=name, description=description, requirements=requirements) as pipe:
            pipe.dag = DAG.from_dict(data["pipeline"])
        return pipe

    @property
    def _cache_location(self) -> "_CacheLocation":
        """Dictionary containing the object that will stored and the location,
        whether it is a filename or a folder.

        Returns:
            Path: Filenames where the pipeline content will be serialized.
        """
        folder = self._cache_dir / self.name / self._create_signature()
        return {
            "base": folder,
            "pipeline": folder / "pipeline.yaml",
            "batch_manager": folder / "batch_manager.json",
            "data": folder / "data",
            "batch_input_data": folder / "batch_input_data",
            "log_file": folder / "pipeline.log",
            "stages_file": folder / "stages.json",
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

        self._save_stages_status()

        self._logger.debug("Pipeline and batch manager saved to cache.")

    def _save_stages_status(self) -> None:
        """Saves the stages status to cache."""
        self.save(
            path=self._cache_location["stages_file"],
            format="json",
            dump={
                "current_stage": self._current_stage,
                "stages_last_batch": self._stages_last_batch,
            },
        )

    def _load_stages_status(self, use_cache: bool = True) -> None:
        """Try to load the stages status from cache, or initialize it if cache file doesn't
        exist or cache is not going to be used."""
        if use_cache and self._cache_location["stages_file"].exists():
            stages_status = read_json(self._cache_location["stages_file"])
            self._current_stage = stages_status["current_stage"]
            self._stages_last_batch = stages_status["stages_last_batch"]
        else:
            self._current_stage = 0
            self._stages_last_batch = [
                [] for _ in range(len(self.dag.get_steps_load_stages()[0]))
            ]

    def _refresh_pipeline_from_cache(self) -> None:
        """Refresh the DAG (and its steps) from the cache file. This is useful as some
        `Step`s can update and change their state during the pipeline execution, and this
        method will make sure the pipeline is up-to-date with the latest changes when
        the pipeline is reloaded from cache.
        """

        def recursively_handle_secrets_and_excluded_attributes(
            cached_model: "BaseModel", model: "BaseModel"
        ) -> None:
            """Recursively handle the secrets and excluded attributes of a `BaseModel`,
            setting the values of the cached model to the values of the model.

            Args:
                cached_model: The cached model that will be updated as it doesn't contain
                    the secrets and excluded attributes (not serialized).
                model: The model that contains the secrets and excluded attributes because
                    it comes from pipeline instantiation.
            """
            for field_name, field_info in cached_model.model_fields.items():
                if field_name in ("pipeline"):
                    continue

                inner_type = extract_annotation_inner_type(field_info.annotation)
                if is_type_pydantic_secret_field(inner_type) or field_info.exclude:
                    setattr(cached_model, field_name, getattr(model, field_name))
                elif isclass(inner_type) and issubclass(inner_type, BaseModel):
                    recursively_handle_secrets_and_excluded_attributes(
                        getattr(cached_model, field_name),
                        getattr(model, field_name),
                    )

        if self._cache_location["pipeline"].exists():
            cached_dag = self.from_yaml(self._cache_location["pipeline"]).dag

            for step_name in cached_dag:
                step_cached: "_Step" = cached_dag.get_step(step_name)[
                    constants.STEP_ATTR_NAME
                ]
                step: "_Step" = self.dag.get_step(step_name)[constants.STEP_ATTR_NAME]
                recursively_handle_secrets_and_excluded_attributes(step_cached, step)

            self.dag = cached_dag

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
        buffer_data_path = self._cache_location["data"] / constants.STEPS_OUTPUTS_PATH
        self._logger.info(f"📝 Pipeline data will be written to '{buffer_data_path}'")
        self._write_buffer = _WriteBuffer(buffer_data_path, self.dag.leaf_steps)

    def _print_load_stages_info(self) -> None:
        """Prints the information about the load stages."""
        stages, _ = self.dag.get_steps_load_stages()
        msg = ""
        for stage, steps in enumerate(stages):
            msg += f"\n * Stage {stage}: {steps}"
        self._logger.info(
            f"⌛ The steps of the pipeline will be loaded in stages:{msg}"
        )

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
        if not self._initialize_pipeline_execution():
            return

        while self._should_continue_processing():  # type: ignore
            self._logger.debug("Waiting for output batch from step...")
            if (batch := self._output_queue.get()) is None:
                self._logger.debug("Received `None` from output queue. Breaking loop.")
                break

            self._logger.debug(
                f"Received batch with seq_no {batch.seq_no} from step '{batch.step_name}'"
                f" from output queue: {batch}"
            )

            self._process_batch(batch)

            # If `_stop_called` was set to `True` while waiting for the output queue, then
            # we need to handle the stop of the pipeline and break the loop to avoid
            # propagating the batches through the pipeline and making the stop process
            # slower.
            with self._stop_called_lock:
                if self._stop_called:
                    self._handle_batch_on_stop(batch)
                    break

            # If there is another load stage and all the `last_batch`es from the stage
            # have been received, then load the next stage.
            if self._should_load_next_stage():
                if not self._update_stage():
                    break

            self._manage_batch_flow(batch)

        self._finalize_pipeline_execution()

    def _initialize_pipeline_execution(self) -> bool:
        """Load the steps of the required stage to initialize the pipeline execution,
        and requests the initial batches to trigger the batch flowing in the pipeline.

        Returns:
            `True` if initialization went OK, `False` otherwise.
        """
        # Wait for all the steps to be loaded correctly
        if not self._run_stage_steps_and_wait(stage=self._current_stage):
            self._set_steps_not_loaded_exception()
            return False

        # Send the "first" batches to the steps so the batches starts flowing through
        # the input queues and output queue
        self._request_initial_batches()

        return True

    def _should_continue_processing(self) -> bool:
        """Condition for the consume batches from the `output_queue` loop.

        Returns:
            `True` if should continue consuming batches, `False` otherwise and the pipeline
            should stop.
        """
        with self._stop_called_lock:
            return self._batch_manager.can_generate() and not self._stop_called  # type: ignore

    def _process_batch(
        self, batch: "_Batch", send_last_batch_flag: bool = True
    ) -> None:
        """Process a batch consumed from the `output_queue`.

        Args:
            batch: the batch to be processed.
        """
        if batch.data_path:
            self._logger.debug(
                f"Reading {batch.seq_no} batch data from '{batch.step_name}': '{batch.data_path}'"
            )
            batch.read_batch_data_from_fs()

        if batch.step_name in self.dag.leaf_steps:
            self._write_buffer.add_batch(batch)  # type: ignore

        if batch.last_batch:
            self._register_stages_last_batch(batch)

            # Make sure to send the `LAST_BATCH_SENT_FLAG` to the predecessors of the step
            # if the batch is the last one, so they stop their processing loop even if they
            # haven't received the last batch because of the routing function.
            if send_last_batch_flag:
                for step_name in self.dag.get_step_predecessors(batch.step_name):
                    if self._is_step_running(step_name):
                        self._send_last_batch_flag_to_step(step_name)

    def _set_step_for_recovering_offline_batch_generation(
        self, step: "_Step", data: List[List[Dict[str, Any]]]
    ) -> None:
        """Sets the required information to recover a pipeline execution from a `_Step`
        that used an `LLM` with offline batch generation.

        Args:
            step: The `_Step` that used an `LLM` with offline batch generation.
            data: The data that was used to generate the batches for the step.
        """
        # Replace step so the attribute `jobs_ids` of the `LLM` is not lost, as it was
        # updated in the child process but not in the main process.
        step_name: str = step.name  # type: ignore
        self.dag.set_step_attr(
            name=step_name, attr=constants.STEP_ATTR_NAME, value=step
        )
        self._recover_offline_batch_generate_for_step = (step_name, data)

    def _add_batch_for_recovering_offline_batch_generation(self) -> None:
        """Adds a dummy `_Batch` to the specified step name (it's a `Task` that used an
        `LLM` with offline batch generation) to recover the pipeline state for offline
        batch generation in next pipeline executions."""
        assert self._batch_manager, "Batch manager is not set"

        if self._recover_offline_batch_generate_for_step is None:
            return

        step_name, data = self._recover_offline_batch_generate_for_step
        self._logger.debug(
            f"Adding batch to '{step_name}' step to recover pipeline execution for offline"
            " batch generation..."
        )
        self._batch_manager.add_batch_to_recover_offline_batch_generation(
            to_step=step_name,
            data=data,
        )

    def _register_stages_last_batch(self, batch: "_Batch") -> None:
        """Registers the last batch received from a step in the `_stages_last_batch`
        dictionary.

        Args:
            batch: The last batch received from a step.
        """
        _, stages_last_steps = self.dag.get_steps_load_stages()
        stage_last_steps = stages_last_steps[self._current_stage]
        if batch.step_name in stage_last_steps:
            self._stages_last_batch[self._current_stage].append(batch.step_name)
            self._stages_last_batch[self._current_stage].sort()

    def _update_stage(self) -> bool:
        """Checks if the steps of next stage should be loaded and updates `_current_stage`
        attribute.

        Returns:
            `True` if updating the stage went OK, `False` otherwise.
        """
        self._current_stage += 1
        if not self._run_stage_steps_and_wait(stage=self._current_stage):
            self._set_steps_not_loaded_exception()
            return False

        return True

    def _should_load_next_stage(self) -> bool:
        """Returns if the next stage should be loaded.

        Returns:
            `True` if the next stage should be loaded, `False` otherwise.
        """
        _, stage_last_steps = self.dag.get_steps_load_stages()
        there_is_next_stage = self._current_stage + 1 < len(stage_last_steps)
        stage_last_batches_received = (
            self._stages_last_batch[self._current_stage]
            == stage_last_steps[self._current_stage]
        )
        return there_is_next_stage and stage_last_batches_received

    def _finalize_pipeline_execution(self) -> None:
        """Finalizes the pipeline execution handling the prematurely stop of the pipeline
        if required, caching the data and ensuring that all the steps finish its execution."""

        # Send `None` to steps `input_queue`s just in case some step is still waiting
        self._notify_steps_to_stop()

        for step_name in self.dag:
            while self._is_step_running(step_name):
                self._logger.debug(f"Waiting for step '{step_name}' to finish...")
                time.sleep(0.5)

        with self._stop_called_lock:
            if self._stop_called:
                self._handle_stop()

            # Reset flag state
            self._stop_called = False

        self._add_batch_for_recovering_offline_batch_generation()

        self._cache()

    def _run_load_queue_loop_in_thread(self) -> threading.Thread:
        """Runs a background thread that reads from the `load_queue` to update the status
        of the number of replicas loaded for each step.

        Returns:
            The thread that was started.
        """
        thread = threading.Thread(target=self._run_load_queue_loop)
        thread.start()
        return thread

    def _run_load_queue_loop(self) -> None:
        """Runs a loop that reads from the `load_queue` to update the status of the number
        of replicas loaded for each step."""

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
                    f"Step '{step_name}' loaded replicas: {self._steps_load_status[step_name]}"
                )

    def _is_step_running(self, step_name: str) -> bool:
        """Checks if the step is running (at least one replica is running).

        Args:
            step_name: The step to be check if running.

        Returns:
            `True` if the step is running, `False` otherwise.
        """
        with self._steps_load_status_lock:
            return self._steps_load_status[step_name] >= 1

    def _run_stage_steps_and_wait(self, stage: int) -> bool:
        """Runs the steps of the specified stage and waits for them to be ready.

        Args:
            stage: the stage from which the steps have to be loaded.

        Returns:
            `True` if all the steps have been loaded correctly, `False` otherwise.
        """

        steps_stages, _ = self.dag.get_steps_load_stages()
        steps = steps_stages[stage]

        # Run the steps of the stage
        self._run_steps(steps=steps)

        # Wait for them to be ready
        self._logger.info(f"⏳ Waiting for all the steps of stage {stage} to load...")
        previous_message = None
        with self._stop_called_lock:
            while not self._stop_called:
                with self._steps_load_status_lock:
                    filtered_steps_load_status = {
                        step_name: replicas
                        for step_name, replicas in self._steps_load_status.items()
                        if step_name in steps
                    }
                    self._logger.debug(
                        f"Steps from stage {stage} loaded: {filtered_steps_load_status}"
                    )

                    if any(
                        replicas_loaded == _STEP_LOAD_FAILED_CODE
                        for replicas_loaded in filtered_steps_load_status.values()
                    ):
                        self._logger.error(
                            f"❌ Failed to load all the steps of stage {stage}"
                        )
                        return False

                    num_steps_loaded = 0
                    replicas_message = ""
                    for step_name, replicas in filtered_steps_load_status.items():
                        step_replica_count = self.dag.get_step_replica_count(step_name)
                        if replicas == step_replica_count:
                            num_steps_loaded += 1
                        replicas_message += f"\n * '{step_name}' replicas: {max(0, replicas)}/{step_replica_count}"

                    message = f"⏳ Steps from stage {stage} loaded: {num_steps_loaded}/{len(filtered_steps_load_status)}{replicas_message}"
                    if num_steps_loaded > 0 and message != previous_message:
                        self._logger.info(message)
                        previous_message = message

                    if num_steps_loaded == len(filtered_steps_load_status):
                        self._logger.info(
                            f"✅ All the steps from stage {stage} have been loaded!"
                        )
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

        if self._should_load_next_stage():
            self._current_stage += 1

    def _wait_step_input_queue_empty(self, step_name: str) -> Union["Queue[Any]", None]:
        """Waits for the input queue of a step to be empty.

        Args:
            step_name: The name of the step.

        Returns:
            The input queue of the step if it's not loaded or finished, `None` otherwise.
        """
        if self._check_step_not_loaded_or_finished(step_name):
            return None

        if input_queue := self.dag.get_step(step_name).get(
            constants.INPUT_QUEUE_ATTR_NAME
        ):
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
            num_replicas = self._steps_load_status[step_name]

            # The step has finished (replicas = 0) or it has failed to load
            if num_replicas in [0, _STEP_LOAD_FAILED_CODE]:
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
        self.dag.set_step_attr(step_name, constants.INPUT_QUEUE_ATTR_NAME, input_queue)
        return input_queue

    @abstractmethod
    def _run_step(self, step: "_Step", input_queue: "Queue[Any]", replica: int) -> None:
        """Runs the `Step` instance.

        Args:
            step: The `Step` instance to run.
            input_queue: The input queue where the step will receive the batches.
            replica: The replica ID assigned.
        """
        pass

    def _run_steps(self, steps: Iterable[str]) -> None:
        """Runs the `Step`s of the pipeline, creating first an input queue for each step
        that will be used to send the batches.

        Args:
            steps:
        """
        for step_name in steps:
            step: "Step" = self.dag.get_step(step_name)[constants.STEP_ATTR_NAME]
            input_queue = self._create_step_input_queue(step_name=step_name)

            # Set `pipeline` to `None` as in some Python environments the pipeline is not
            # picklable and it will raise an error when trying to send the step to the process.
            # `TypeError: cannot pickle 'code' object`
            step.pipeline = None

            if not step.is_normal and step.resources.replicas > 1:  # type: ignore
                self._logger.warning(
                    f"Step '{step_name}' is a `GeneratorStep` or `GlobalStep` and has more"
                    " than 1 replica. Only `Step` instances can have more than 1 replica."
                    " The number of replicas for the step will be set to 1."
                )

            step_num_replicas: int = step.resources.replicas if step.is_normal else 1  # type: ignore
            for replica in range(step_num_replicas):
                self._logger.debug(
                    f"Running 1 replica of step '{step.name}' with ID {replica}..."
                )
                self._run_step(
                    step=step.model_copy(deep=True),
                    input_queue=input_queue,
                    replica=replica,
                )

    def _add_batches_back_to_batch_manager(self) -> None:
        """Add the `Batch`es that were sent to a `Step` back to the `_BatchManager`. This
        method should be used when the pipeline has been stopped prematurely."""
        for step_name in self.dag:
            node = self.dag.get_step(step_name)
            step: "_Step" = node[constants.STEP_ATTR_NAME]
            if step.is_generator:
                continue
            if input_queue := node.get(constants.INPUT_QUEUE_ATTR_NAME):
                while not input_queue.empty():
                    batch = input_queue.get()
                    if not isinstance(batch, _Batch):
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
            self._process_batch(batch, send_last_batch_flag=False)
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

        self._register_batch(batch)

        route_to, do_not_route_to, routed = self._get_successors(batch)

        # Keep track of the steps that the batch was routed to
        if routed:
            batch.batch_routed_to = route_to

        self._set_next_expected_seq_no(
            steps=do_not_route_to,
            from_step=batch.step_name,
            next_expected_seq_no=batch.seq_no + 1,
        )

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
            while new_batch := self._batch_manager.get_batch(successor):
                self._send_batch_to_step(new_batch)

        if not step.is_generator:
            # Step ("this", the one from which the batch was received) has enough data on its
            # buffers to create a new batch
            while new_batch := self._batch_manager.get_batch(step.name):  # type: ignore
                # if new_batch := self._batch_manager.get_batch(step.name):  # type: ignore
                self._send_batch_to_step(new_batch)
            else:
                self._request_more_batches_if_needed(step)
        else:
            if len(self.dag) == 1:
                self._request_batch_from_generator(step.name)  # type: ignore

        self._cache()

    def _send_to_step(self, step_name: str, to_send: Any) -> None:
        """Sends something to the input queue of a step.

        Args:
            step_name: The name of the step.
            to_send: The object to send.
        """
        input_queue = self.dag.get_step(step_name)[constants.INPUT_QUEUE_ATTR_NAME]
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

        step: "_Step" = self.dag.get_step(batch.step_name)[constants.STEP_ATTR_NAME]
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

    def _gather_requirements(self) -> List[str]:
        """Extracts the requirements from the steps to be used in the pipeline.

        Returns:
            List of requirements gathered from the steps.
        """
        steps_requirements = []
        for step in self.dag:
            step_req = self.dag.get_step(step)[constants.STEP_ATTR_NAME].requirements
            steps_requirements.extend(step_req)

        return steps_requirements

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
        self._logger.debug(
            f"Sending `LAST_BATCH_SENT_FLAG` to '{step_name}' step to stop processing"
            " batches..."
        )

        for _ in range(self.dag.get_step_replica_count(step_name)):
            self._send_to_step(step_name, constants.LAST_BATCH_SENT_FLAG)
        self._batch_manager.set_last_batch_flag_sent_to(step_name)  # type: ignore

    def _request_initial_batches(self) -> None:
        """Requests the initial batches to the generator steps."""
        assert self._batch_manager, "Batch manager is not set"

        for step in self._batch_manager._steps.values():
            if not self._is_step_running(step.step_name):
                continue
            if batch := step.get_batch():
                self._logger.debug(
                    f"Sending initial batch to '{step.step_name}' step: {batch}"
                )
                self._send_batch_to_step(batch)

        for step_name in self.dag.root_steps:
            if not self._is_step_running(step_name):
                continue
            seq_no = 0
            if last_batch := self._batch_manager.get_last_batch(step_name):
                seq_no = last_batch.seq_no + 1
            batch = _Batch(seq_no=seq_no, step_name=step_name, last_batch=self._dry_run)
            self._logger.debug(
                f"Requesting initial batch to '{step_name}' generator step: {batch}"
            )
            self._send_batch_to_step(batch)

    def _request_batch_from_generator(self, step_name: str) -> None:
        """Request a new batch to a `GeneratorStep`.

        Args:
            step_name: the name of the `GeneratorStep` to which a batch has to be requested.
        """
        # Get the last batch that the previous step sent to generate the next batch
        # (next `seq_no`).
        last_batch = self._batch_manager.get_last_batch_sent(step_name)  # type: ignore
        if last_batch is None:
            return
        self._send_batch_to_step(last_batch.next_batch())

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

            self._request_batch_from_generator(previous_step_name)

    def _handle_batch_on_stop(self, batch: "_Batch") -> None:
        """Handles a batch that was received from the output queue when the pipeline was
        stopped. It will add and register the batch in the batch manager.

        Args:
            batch: The batch to handle.
        """
        assert self._batch_manager, "Batch manager is not set"

        self._batch_manager.register_batch(batch)
        step: "Step" = self.dag.get_step(batch.step_name)[constants.STEP_ATTR_NAME]
        for successor in self.dag.get_step_successors(step.name):  # type: ignore
            self._batch_manager.add_batch(successor, batch)

    def _get_step_from_batch(self, batch: "_Batch") -> "Step":
        """Gets the `Step` instance from a batch.

        Args:
            batch: The batch to get the step from.

        Returns:
            The `Step` instance.
        """
        return self.dag.get_step(batch.step_name)[constants.STEP_ATTR_NAME]

    def _notify_steps_to_stop(self) -> None:
        """Notifies the steps to stop their infinite running loop by sending `None` to
        their input queues."""
        with self._steps_load_status_lock:
            for step_name, replicas in self._steps_load_status.items():
                if replicas > 0:
                    self._send_to_step(step_name, None)

    def _get_successors(self, batch: "_Batch") -> Tuple[List[str], List[str], bool]:
        """Gets the successors and the successors to which the batch has to be routed.

        Args:
            batch: The batch to which the successors will be determined.

        Returns:
            The successors to route the batch to and whether the batch was routed using
            a routing function.
        """
        node = self.dag.get_step(batch.step_name)
        step: "Step" = node[constants.STEP_ATTR_NAME]
        successors = list(self.dag.get_step_successors(step.name))  # type: ignore
        route_to = successors

        # Check if the step has a routing function to send the batch to specific steps
        if routing_batch_function := node.get(
            constants.ROUTING_BATCH_FUNCTION_ATTR_NAME
        ):
            route_to = routing_batch_function(batch, successors)
            successors_str = ", ".join(f"'{successor}'" for successor in route_to)
            self._logger.info(
                f"🚏 Using '{step.name}' routing function to send batch {batch.seq_no} to steps: {successors_str}"
            )

        return route_to, list(set(successors) - set(route_to)), route_to != successors

    def _set_next_expected_seq_no(
        self, steps: List[str], from_step: str, next_expected_seq_no: int
    ) -> None:
        """Sets the next expected sequence number of a `_Batch` received by `step` from
        `from_step`. This is necessary as some `Step`s might not receive all the batches
        comming from the previous steps because there is a routing batch function.

        Args:
            steps: list of steps to which the next expected sequence number of a `_Batch`
                from `from_step` has to be updated in the `_BatchManager`.
            from_step: the name of the step from which the next expected sequence number
                of a `_Batch` has to be updated in `steps`.
            next_expected_seq_no: the number of the next expected sequence number of a `Batch`
                from `from_step`.
        """
        assert self._batch_manager, "Batch manager is not set"

        for step in steps:
            self._batch_manager.set_next_expected_seq_no(
                step_name=step,
                from_step=from_step,
                next_expected_seq_no=next_expected_seq_no,
            )

    @abstractmethod
    def _teardown(self) -> None:
        """Clean/release/stop resources reserved to run the pipeline."""
        pass

    @abstractmethod
    def _set_steps_not_loaded_exception(self) -> None:
        """Used to raise `RuntimeError` when the load of the steps failed.

        Raises:
            RuntimeError: containing the information and why a step failed to be loaded.
        """
        pass

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


def set_pipeline_running_env_variables(
    pipeline_name: str, pipeline_cache_id: str, pipeline_cache_dir: str
) -> None:
    os.environ[constants.PIPELINE_NAME_ENV_NAME] = pipeline_name
    os.environ[constants.PIPELINE_CACHE_ID_ENV_NAME] = pipeline_cache_id
    os.environ[constants.PIPELINE_CACHE_DIR_ENV_NAME] = pipeline_cache_dir
