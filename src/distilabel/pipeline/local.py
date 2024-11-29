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

import multiprocessing as mp
import signal
import sys
from multiprocessing.pool import Pool
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    cast,
)

import tblib

from distilabel.constants import SIGINT_HANDLER_CALLED_ENV_NAME
from distilabel.distiset import create_distiset
from distilabel.exceptions import DistilabelOfflineBatchGenerationNotFinishedException
from distilabel.pipeline.base import BasePipeline, set_pipeline_running_env_variables
from distilabel.pipeline.ray import RayPipeline
from distilabel.pipeline.step_wrapper import _StepWrapper, _StepWrapperException
from distilabel.utils.logging import setup_logging, stop_logging
from distilabel.utils.ray import script_executed_in_ray_cluster

if TYPE_CHECKING:
    import logging
    from queue import Queue

    from distilabel.distiset import Distiset
    from distilabel.pipeline.typing import InputDataset
    from distilabel.steps.base import _Step


_SUBPROCESS_EXCEPTION: Union[Exception, None] = None


def _init_worker(
    log_queue: "Queue[Any]", pipeline_name: str, pipeline_cache_id: str
) -> None:
    """Init function for the child processes that will execute the `Step`s of the `Pipeline`.

    Args:
        log_queue: The queue to send the logs to the main process.
    """

    # Register a signal handler for SIGINT to avoid the default behavior of the process
    # to terminate when the parent process receives a SIGINT signal. Instead, set an env
    # variable when SIGINT is received. Child process can check the value of this env
    # variable in sections of the code where they need to stop the execution if SIGINT
    # was received (such as offline batch generation polling).
    def signal_handler(sig: int, frame: Any) -> None:
        import os

        os.environ[SIGINT_HANDLER_CALLED_ENV_NAME] = "1"

    signal.signal(signal.SIGINT, signal_handler)
    set_pipeline_running_env_variables(pipeline_name, pipeline_cache_id)
    setup_logging(log_queue)


# We create a custom `Pool` class so the created processes are not daemons, allowing
# them to create child processes if necessary (for example when using `vLLM` with `tensor_parallel_size`)
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class _NoDaemonProcess(mp.Process):
    @property
    def daemon(self) -> bool:
        return False

    @daemon.setter
    def daemon(self, value: bool) -> None:  # type: ignore
        pass


class _NoDaemonContext(type(mp.get_context())):
    Process = _NoDaemonProcess


class _NoDaemonPool(Pool):
    def __init__(
        self,
        processes: Union[int, None] = None,
        initializer: Union[Callable[..., object], None] = None,
        initargs: Iterable[Any] = ...,  # type: ignore
        maxtasksperchild: Union[int, None] = None,
    ) -> None:
        super().__init__(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild,
            context=_NoDaemonContext(),  # type: ignore
        )


class Pipeline(BasePipeline):
    """Local pipeline implementation using `multiprocessing`."""

    def ray(
        self,
        ray_head_node_url: Optional[str] = None,
        ray_init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> RayPipeline:
        """Creates a `RayPipeline` using the init parameters of this pipeline. This is a
        convenient method that can be used to "transform" one common `Pipeline` to a `RayPipeline`
        and it's mainly used by the CLI.

        Args:
            ray_head_node_url: The URL that can be used to connect to the head node of
                the Ray cluster. Normally, you won't want to use this argument as the
                recommended way to submit a job to a Ray cluster is using the [Ray Jobs
                CLI](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html#ray-jobs-overview).
                Defaults to `None`.
            ray_init_kwargs: kwargs that will be passed to the `ray.init` method. Defaults
                to `None`.

        Returns:
            A `RayPipeline` instance.
        """
        pipeline = RayPipeline(
            name=self.name,
            description=self.description,
            cache_dir=self._cache_dir,
            enable_metadata=self._enable_metadata,
            requirements=self.requirements,
            ray_head_node_url=ray_head_node_url,
            ray_init_kwargs=ray_init_kwargs,
        )
        pipeline.dag = self.dag
        return pipeline

    def run(
        self,
        parameters: Optional[Dict[Any, Dict[str, Any]]] = None,
        load_groups: Optional[List[List[Any]]] = None,
        use_cache: bool = True,
        storage_parameters: Optional[Dict[str, Any]] = None,
        use_fs_to_pass_data: bool = False,
        dataset: Optional["InputDataset"] = None,
        dataset_batch_size: int = 50,
        logging_handlers: Optional[List["logging.Handler"]] = None,
    ) -> "Distiset":
        """Runs the pipeline.

        Args:
            parameters: A dictionary with the step name as the key and a dictionary with
                the runtime parameters for the step as the value. Defaults to `None`.
            load_groups: A list containing list of steps that has to be loaded together
                and in isolation with respect to the rest of the steps of the pipeline.
                Defaults to `None`.
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
            dataset_batch_size: if `dataset` is given, this will be the size of the batches
                yield by the `GeneratorStep` created using the `dataset`. Defaults to `50`.
            logging_handlers: A list of logging handlers that will be used to log the
                output of the pipeline. This argument can be useful so the logging messages
                can be extracted and used in a different context. Defaults to `None`.

        Returns:
            The `Distiset` created by the pipeline.

        Raises:
            RuntimeError: If the pipeline fails to load all the steps.
        """
        if script_executed_in_ray_cluster():
            print("Script running in Ray cluster... Using `RayPipeline`...")
            return self.ray().run(
                parameters=parameters,
                use_cache=use_cache,
                storage_parameters=storage_parameters,
                use_fs_to_pass_data=use_fs_to_pass_data,
                dataset=dataset,
                dataset_batch_size=dataset_batch_size,
            )

        self._log_queue = cast("Queue[Any]", mp.Queue())

        if distiset := super().run(
            parameters=parameters,
            load_groups=load_groups,
            use_cache=use_cache,
            storage_parameters=storage_parameters,
            use_fs_to_pass_data=use_fs_to_pass_data,
            dataset=dataset,
            dataset_batch_size=dataset_batch_size,
            logging_handlers=logging_handlers,
        ):
            return distiset

        num_processes = self.dag.get_total_replica_count()
        with (
            mp.Manager() as manager,
            _NoDaemonPool(
                num_processes,
                initializer=_init_worker,
                initargs=(
                    self._log_queue,
                    self.name,
                    self.signature,
                ),
            ) as pool,
        ):
            self._manager = manager
            self._pool = pool
            self._output_queue = self.QueueClass()
            self._load_queue = self.QueueClass()
            self._handle_keyboard_interrupt()

            # Run the loop for receiving the load status of each step
            self._load_steps_thread = self._run_load_queue_loop_in_thread()

            # Start a loop to receive the output batches from the steps
            self._output_queue_thread = self._run_output_queue_loop_in_thread()
            self._output_queue_thread.join()

            self._teardown()

            if self._exception:
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

    @property
    def QueueClass(self) -> Callable:
        """The callable used to create the input and output queues.

        Returns:
            The callable to create a `Queue`.
        """
        assert self._manager, "Manager is not initialized"
        return self._manager.Queue

    def _run_step(self, step: "_Step", input_queue: "Queue[Any]", replica: int) -> None:
        """Runs the `Step` wrapped in a `_ProcessWrapper` in a separate process of the
        `Pool`.

        Args:
            step: The step to run.
            input_queue: The input queue to send the data to the step.
            replica: The replica ID assigned.
        """
        assert self._pool, "Pool is not initialized"

        step_wrapper = _StepWrapper(
            step=step,  # type: ignore
            replica=replica,
            input_queue=input_queue,
            output_queue=self._output_queue,
            load_queue=self._load_queue,
            dry_run=self._dry_run,
            ray_pipeline=False,
        )

        self._pool.apply_async(step_wrapper.run, error_callback=self._error_callback)

    def _error_callback(self, e: BaseException) -> None:
        """Error callback that will be called when an error occurs in a `Step` process.

        Args:
            e: The exception raised by the process.
        """
        global _SUBPROCESS_EXCEPTION

        # First we check that the exception is a `_StepWrapperException`, otherwise, we
        # print it out and stop the pipeline, since some errors may be unhandled
        if not isinstance(e, _StepWrapperException):
            self._logger.error(f"❌ Failed with an unhandled exception: {e}")
            self._stop()
            return

        if e.is_load_error:
            self._logger.error(f"❌ Failed to load step '{e.step.name}': {e.message}")
            _SUBPROCESS_EXCEPTION = e.subprocess_exception
            _SUBPROCESS_EXCEPTION.__traceback__ = tblib.Traceback.from_string(  # type: ignore
                e.formatted_traceback
            ).as_traceback()
            return

        # If the step is global, is not in the last trophic level and has no successors,
        # then we can ignore the error and continue executing the pipeline
        step_name: str = e.step.name  # type: ignore
        if (
            e.step.is_global
            and not self.dag.step_in_last_trophic_level(step_name)
            and list(self.dag.get_step_successors(step_name)) == []
        ):
            self._logger.error(
                f"✋ An error occurred when running global step '{step_name}' with no"
                " successors and not in the last trophic level. Pipeline execution can"
                f" continue. Error will be ignored."
            )
            self._logger.error(f"Subprocess traceback:\n\n{e.formatted_traceback}")
            return

        # Handle tasks using an `LLM` using offline batch generation
        if isinstance(
            e.subprocess_exception, DistilabelOfflineBatchGenerationNotFinishedException
        ):
            self._logger.info(
                f"⏹️ '{e.step.name}' task stopped pipeline execution: LLM offline batch"
                " generation in progress. Rerun pipeline with cache to check results and"
                " continue execution."
            )
            self._set_step_for_recovering_offline_batch_generation(e.step, e.data)  # type: ignore
            with self._stop_called_lock:
                if not self._stop_called:
                    self._stop(acquire_lock=False)
            return

        # Global step with successors failed
        self._logger.error(f"An error occurred in global step '{step_name}'")
        self._logger.error(f"Subprocess traceback:\n\n{e.formatted_traceback}")

        self._stop()

    def _teardown(self) -> None:
        """Clean/release/stop resources reserved to run the pipeline."""
        if self._write_buffer:
            self._write_buffer.close()

        if self._batch_manager:
            self._batch_manager = None

        self._stop_load_queue_loop()
        self._load_steps_thread.join()

        if self._pool:
            self._pool.terminate()
            self._pool.join()

        if self._manager:
            self._manager.shutdown()
            self._manager.join()

    def _set_steps_not_loaded_exception(self) -> None:
        """Raises a `RuntimeError` notifying that the steps load has failed.

        Raises:
            RuntimeError: containing the information and why a step failed to be loaded.
        """
        self._exception = RuntimeError(
            "Failed to load all the steps. Could not run pipeline."
        )
        self._exception.__cause__ = _SUBPROCESS_EXCEPTION

    def _stop(self, acquire_lock: bool = True) -> None:
        """Stops the pipeline execution. It will first send `None` to the input queues
        of all the steps and then wait until the output queue is empty i.e. all the steps
        finished processing the batches that were sent before the stop flag. Then it will
        send `None` to the output queue to notify the pipeline to stop.

        Args:
            acquire_lock: Whether to acquire the lock to access the `_stop_called` attribute.
        """

        if acquire_lock:
            self._stop_called_lock.acquire()

        if self._stop_called:
            self._stop_calls += 1
            if self._stop_calls == 1:
                self._logger.warning("🛑 Press again to force the pipeline to stop.")
            elif self._stop_calls > 1:
                self._logger.warning("🛑 Forcing pipeline interruption.")

                if self._pool:
                    self._pool.terminate()
                    self._pool.join()
                    self._pool = None

                if self._manager:
                    self._manager.shutdown()
                    self._manager.join()
                    self._manager = None

                stop_logging()

                sys.exit(1)

            return
        self._stop_called = True

        if acquire_lock:
            self._stop_called_lock.release()

        self._logger.debug(
            f"Steps loaded before calling `stop`: {self._steps_load_status}"
        )
        self._logger.info(
            "🛑 Stopping pipeline. Waiting for steps to finish processing batches..."
        )

        self._stop_output_queue_loop()
