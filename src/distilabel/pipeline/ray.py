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

import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from distilabel.constants import INPUT_QUEUE_ATTR_NAME, STEP_ATTR_NAME
from distilabel.distiset import create_distiset
from distilabel.errors import DistilabelUserError
from distilabel.llms.vllm import vLLM
from distilabel.pipeline.base import BasePipeline, set_pipeline_running_env_variables
from distilabel.pipeline.step_wrapper import _StepWrapper
from distilabel.utils.logging import setup_logging, stop_logging
from distilabel.utils.serialization import TYPE_INFO_KEY

if TYPE_CHECKING:
    from os import PathLike
    from queue import Queue

    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from distilabel.distiset import Distiset
    from distilabel.pipeline.typing import InputDataset
    from distilabel.steps.base import _Step


class RayPipeline(BasePipeline):
    """Ray pipeline implementation allowing to run a pipeline in a Ray cluster."""

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        cache_dir: Optional[Union[str, "PathLike"]] = None,
        enable_metadata: bool = False,
        requirements: Optional[List[str]] = None,
        ray_head_node_url: Optional[str] = None,
        ray_init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the `RayPipeline` instance.

        Args:
            name: The name of the pipeline.
            description: A description of the pipeline. Defaults to `None`.
            cache_dir: A directory where the pipeline will be cached. Defaults to `None`.
            enable_metadata: Whether to include the distilabel metadata column for the pipeline
                in the final `Distiset`. It contains metadata used by distilabel, for example
                the raw outputs of the `LLM` without processing would be here, inside `raw_output_...`
                field. Defaults to `False`.
            requirements: List of requirements that must be installed to run the Pipeline.
                Defaults to `None`, but can be helpful to inform in a pipeline to be shared
                that this requirements must be installed.
            ray_head_node_url: The URL that can be used to connect to the head node of
                the Ray cluster. Normally, you won't want to use this argument as the
                recommended way to submit a job to a Ray cluster is using the [Ray Jobs
                CLI](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html#ray-jobs-overview).
                Defaults to `None`.
            ray_init_kwargs: kwargs that will be passed to the `ray.init` method. Defaults
                to `None`.
        """
        super().__init__(name, description, cache_dir, enable_metadata, requirements)

        self._ray_head_node_url = ray_head_node_url
        self._ray_init_kwargs = ray_init_kwargs or {}
        self._ray_node_ids = {}

    def run(
        self,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache: bool = True,
        storage_parameters: Optional[Dict[str, Any]] = None,
        use_fs_to_pass_data: bool = False,
        dataset: Optional["InputDataset"] = None,
    ) -> "Distiset":
        """Runs the pipeline in the Ray cluster.

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

        Raises:
            RuntimeError: If the pipeline fails to load all the steps.
        """
        self._check_no_llms_using_offline_batch_generation()

        self._init_ray()

        self._log_queue = self.QueueClass(
            actor_options={"name": f"distilabel-{self.name}-log-queue"}
        )

        if distiset := super().run(
            parameters,
            use_cache,
            storage_parameters,
            use_fs_to_pass_data,
            dataset=dataset,
        ):
            return distiset

        self._logger.info(f"Ray nodes GPUs: {self._ray_node_ids}")

        self._output_queue = self.QueueClass(
            actor_options={"name": f"distilabel-{self.name}-output-queue"}
        )
        self._load_queue = self.QueueClass(
            actor_options={"name": f"distilabel-{self.name}-load-queue"}
        )
        self._handle_keyboard_interrupt()

        # Run the loop for receiving the load status of each step
        self._load_steps_thread = self._run_load_queue_loop_in_thread()

        # Start a loop to receive the output batches from the steps
        self._output_queue_thread = self._run_output_queue_loop_in_thread()
        self._output_queue_thread.join()

        self._teardown()

        if self._exception:
            stop_logging()
            raise self._exception

        distiset = create_distiset(
            self._cache_location["data"],
            pipeline_path=self._cache_location["pipeline"],
            log_filename_path=self._cache_location["log_file"],
            enable_metadata=self._enable_metadata,
            dag=self.dag,
        )

        stop_logging()

        return distiset

    def _check_no_llms_using_offline_batch_generation(self) -> None:
        """Checks if there are any `LLM` steps using the `offline_batch_generate` method
        and raises an exception if so. This method is not supported in the Ray pipeline."""
        for step_name in self.dag:
            step: "_Step" = self.dag.get_step(step_name)[STEP_ATTR_NAME]
            if not hasattr(step, "llm"):
                continue
            if step.llm.use_offline_batch_generation:  # type: ignore
                raise DistilabelUserError(
                    f"Step '{step_name}' uses an `LLM` with offline batch generation because"
                    "`use_offline_batch_generation=True`. `LLM`s using this method are not"
                    " supported in the Ray pipeline.",
                    page="sections/how_to_guides/advanced/offline-batch-generation",
                )

    def _init_ray(self) -> None:
        """Inits or connects to a Ray cluster."""
        try:
            import ray
        except ImportError as ie:
            raise ImportError(
                "ray is not installed. Please install it using `pip install ray[default]`."
            ) from ie

        if self._ray_head_node_url:
            ray.init(
                self._ray_head_node_url,
                runtime_env={"pip": self.requirements},
                **self._ray_init_kwargs,
            )
        elif not ray.is_initialized():
            # Init a local Ray cluster
            ray.init(**self._ray_init_kwargs)

        self._ray_node_ids = self._get_ray_gpus_per_node()

    def _get_ray_gpus_per_node(self) -> Dict[str, int]:
        """Gets the number of GPUs per node in the Ray cluster.

        Returns:
            A dictionary in which the keys are the node IDs and the values the number of
                GPUs per node.
        """
        import ray

        gpus_per_node = {}
        for node in ray.nodes():
            node_id = node["NodeID"]
            gpus = int(node["Resources"].get("GPU", 0))
            gpus_per_node[node_id] = gpus
        return gpus_per_node

    @property
    def QueueClass(self) -> Callable:
        from ray.util.queue import Queue

        return Queue

    def _create_step_input_queue(self, step_name: str) -> "Queue[Any]":
        """Creates an input queue for a step. Override to set actor name.

        Args:
            step_name: The name of the step.

        Returns:
            The input queue created.
        """
        input_queue = self.QueueClass(
            actor_options={"name": f"distilabel-{self.name}-input-queue-{step_name}"}
        )
        self.dag.set_step_attr(step_name, INPUT_QUEUE_ATTR_NAME, input_queue)
        return input_queue

    def _run_step(self, step: "_Step", input_queue: "Queue[Any]", replica: int) -> None:
        """Creates a replica of an `Step` using a Ray Actor.

        Args:
            step: The step to run.
            input_queue: The input queue to send the data to the step.
            replica: The replica ID assigned.
        """
        import ray

        @ray.remote
        class _StepWrapperRay:
            def __init__(
                self,
                step_wrapper: _StepWrapper,
                log_queue: "Queue[Any]",
                pipeline_name: str,
                pipeline_cache_id: str,
                pipeline_cache_dir: str,
            ) -> None:
                self._step_wrapper = step_wrapper
                self._log_queue = log_queue
                self._pipeline_name = pipeline_name
                self._pipeline_cache_id = pipeline_cache_id
                self._pipeline_cache_dir = pipeline_cache_dir

            def run(self) -> str:
                setup_logging(log_queue=self._log_queue)
                set_pipeline_running_env_variables(
                    pipeline_name=self._pipeline_name,
                    pipeline_cache_id=self._pipeline_cache_id,
                    pipeline_cache_dir=self._pipeline_cache_dir,
                )
                return self._step_wrapper.run()

        resources: Dict[str, Any] = {
            "name": f"distilabel-{self.name}-{step.name}-{replica}"
        }

        if hasattr(step, "llm") and isinstance(step.llm, vLLM):  # type: ignore
            resources["scheduling_strategy"] = self._create_vllm_placement_group(step)
        else:
            if step.resources.cpus is not None:
                resources["num_cpus"] = step.resources.cpus

            if step.resources.gpus is not None:
                resources["num_gpus"] = step.resources.gpus

            if step.resources.memory is not None:
                resources["memory"] = step.resources.memory

            if step.resources.resources is not None:
                resources["resources"] = step.resources.resources

        _StepWrapperRay = _StepWrapperRay.options(**resources)  # type: ignore

        self._logger.debug(
            f"Creating Ray actor for '{step.name}' (replica ID: {replica}) with resources:"
            f" {resources}"
        )
        step_wrapper = _StepWrapperRay.remote(
            step_wrapper=_StepWrapper(
                step=step,  # type: ignore
                replica=replica,
                input_queue=input_queue,
                output_queue=self._output_queue,
                load_queue=self._load_queue,
                dry_run=self._dry_run,
                ray_pipeline=True,
            ),
            log_queue=self._log_queue,
            pipeline_name=self.name,
            pipeline_cache_id=self._create_signature(),
            pipeline_cache_dir=self._cache_location["base"],
        )

        self._logger.debug(
            f"Executing remote `run` method of Ray actor for '{step.name}' (replica ID:"
            f" {replica})..."
        )
        step_wrapper.run.remote()

    def _create_vllm_placement_group(
        self, step: "_Step"
    ) -> "PlacementGroupSchedulingStrategy":
        """Creates a Ray placement group with as many GPU bundles as `tensor_parallel_size`
        specified in the `vLLM` initialisation. The created placement group uses the `STRICT_PACK`
        strategy if the `pipeline_parallel_size` is less or equal to 1, otherwise it uses
        `SPREAD` (placement group with GPU bundles in several nodes). In addition, the created
        placement group is targeted to be created in a specific node. This avoids having
        `vLLM` raising the exception `Ray does not allocate any GPUs on the driver node...`,
        as it assures that the driver `_StepWrapperRay` actor created resides in the same
        node as the ray actors created by `vLLM` for the distributed inference.

        Args:
            step: the step which uses `vLLM`.

        Returns:
            A `PlacementGroupSchedulingStrategy` using the created `PlacementGroup`.
        """
        import ray

        llm = step.llm  # type: ignore
        tensor_parallel_size = llm.extra_kwargs.get("tensor_parallel_size", 1)  # type: ignore
        pipeline_parallel_size = llm.extra_kwargs.get("pipeline_parallel_size", 1)  # type: ignore

        # Calculate total GPUs needed
        total_gpus_needed = tensor_parallel_size * pipeline_parallel_size

        # Count available GPUs across all nodes
        total_available_gpus = sum(self._ray_node_ids.values())
        self._logger.info(
            f"`vLLM` placement group for '{step.name}' step requires {total_gpus_needed}"
            f" GPUs. Total available GPUs: {total_available_gpus}."
        )

        if total_available_gpus < total_gpus_needed:
            raise ValueError(
                f"Ray cluster does not allocate enough GPUs to create the placement group"
                f" required by the `vLLM` instance of the step '{step.name}'."
                f" Needed: {total_gpus_needed}, Available: {total_available_gpus}"
            )

        # Update the available GPU count
        selected_node_id = None
        gpus_left_needed = total_gpus_needed
        for node_id in self._ray_node_ids:
            gpus_to_allocate = min(self._ray_node_ids[node_id], gpus_left_needed)
            self._ray_node_ids[node_id] -= gpus_to_allocate
            gpus_left_needed -= gpus_to_allocate
            if gpus_left_needed == 0:
                if pipeline_parallel_size == 1:
                    selected_node_id = node_id
                break

        # Create a placement group
        pg = ray.util.placement_group(
            #  # Create `tensor_parallel_size` GPU bundles and at least one CPU bundle
            # so the actors can be scheduled and executed (1 CPU bundle can have infinite actors):
            # https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html#schedule-tasks-and-actors-to-placement-groups-use-reserved-resources
            bundles=[{"CPU": 1.0}] + [{"GPU": 1.0}] * total_gpus_needed,
            strategy="SPREAD" if pipeline_parallel_size > 1 else "STRICT_PACK",
            _soft_target_node_id=selected_node_id,
        )

        self._logger.info(
            f"Step '{step.name}' uses `vLLM`. Created a Ray placement group with bundle"
            f" specs: {pg.bundle_specs}"
        )

        return ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(  # type: ignore
            placement_group=pg,
        )

    def _teardown(self) -> None:
        """Clean/release/stop resources reserved to run the pipeline."""
        if self._write_buffer:
            self._write_buffer.close()

        if self._batch_manager:
            self._batch_manager = None

        self._stop_load_queue_loop()
        self._load_steps_thread.join()

    def _set_steps_not_loaded_exception(self) -> None:
        pass

    def _stop(self) -> None:
        """Stops the pipeline execution. It will first send `None` to the input queues
        of all the steps and then wait until the output queue is empty i.e. all the steps
        finished processing the batches that were sent before the stop flag. Then it will
        send `None` to the output queue to notify the pipeline to stop."""
        with self._stop_called_lock:
            if self._stop_called:
                self._stop_calls += 1
                if self._stop_calls == 1:
                    self._logger.warning(
                        "🛑 Press again to force the pipeline to stop."
                    )
                elif self._stop_calls > 1:
                    self._logger.warning("🛑 Forcing pipeline interruption.")

                    stop_logging()

                    sys.exit(1)

                return
            self._stop_called = True

        self._logger.debug(
            f"Steps loaded before calling `stop`: {self._steps_load_status}"
        )
        self._logger.info(
            "🛑 Stopping pipeline. Waiting for steps to finish processing batches..."
        )

        self._stop_output_queue_loop()

    def dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the pipeline information. Override to hardcode the type info to `Pipeline`,
        as we don't want to create a `RayPipeline` directly but create it using `Pipeline.ray`
        method.

        Returns:
            The pipeline dump.
        """
        from distilabel.pipeline import Pipeline

        dict_ = super().dump()
        dict_["pipeline"][TYPE_INFO_KEY] = {
            "module": Pipeline.__module__,
            "name": Pipeline.__name__,
        }
        return dict_
