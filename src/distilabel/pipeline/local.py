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
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, cast

import tblib

from distilabel.distiset import create_distiset
from distilabel.llms.mixins import CudaDevicePlacementMixin
from distilabel.pipeline.base import (
    BasePipeline,
)
from distilabel.pipeline.batch import _Batch
from distilabel.pipeline.constants import (
    LAST_BATCH_SENT_FLAG,
)
from distilabel.steps.tasks.base import Task
from distilabel.utils.logging import setup_logging

if TYPE_CHECKING:
    from queue import Queue

    from distilabel.distiset import Distiset
    from distilabel.pipeline.typing import StepLoadStatus
    from distilabel.steps.base import GeneratorStep, Step, _Step


_SUBPROCESS_EXCEPTION: Union[Exception, None] = None


def _init_worker(log_queue: "Queue[Any]") -> None:
    """Init function for the child processes that will execute the `Step`s of the `Pipeline`.

    Args:
        log_queue: The queue to send the logs to the main process.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    setup_logging(log_queue)


class Pipeline(BasePipeline):
    """Local pipeline implementation using `multiprocessing`."""

    def run(
        self,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache: bool = True,
        storage_parameters: Optional[Dict[str, Any]] = None,
        use_fs_to_pass_data: bool = False,
    ) -> "Distiset":
        """Runs the pipeline.

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

        Raises:
            RuntimeError: If the pipeline fails to load all the steps.
        """
        log_queue = mp.Queue()

        self._set_logging_parameters(
            {"log_queue": log_queue, "filename": self._cache_location["log_file"]}
        )

        if distiset := super().run(
            parameters, use_cache, storage_parameters, use_fs_to_pass_data
        ):
            return distiset

        num_processes = self.dag.get_total_replica_count()
        ctx = mp.get_context()  # type: ignore
        with ctx.Manager() as manager, ctx.Pool(
            num_processes,
            initializer=_init_worker,
            initargs=(log_queue,),
        ) as pool:
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
        )

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

        process_wrapper = _ProcessWrapper(
            step=step,
            replica=replica,
            input_queue=input_queue,
            output_queue=self._output_queue,
            load_queue=self._load_queue,
            dry_run=self._dry_run,
        )

        self._pool.apply_async(process_wrapper.run, error_callback=self._error_callback)

    def _error_callback(self, e: BaseException) -> None:
        """Error callback that will be called when an error occurs in a `Step` process.

        Args:
            e: The exception raised by the process.
        """
        global _SUBPROCESS_EXCEPTION

        # First we check that the exception is a `_ProcessWrapperException`, otherwise, we
        # print it out and stop the pipeline, since some errors may be unhandled
        if not isinstance(e, _ProcessWrapperException):
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

                    if self._pool:
                        self._pool.terminate()
                        self._pool.join()
                        self._pool = None

                    if self._manager:
                        self._manager.shutdown()
                        self._manager.join()
                        self._manager = None

                    sys.exit(1)

                return
            self._stop_called = True

        self._logger.debug(
            f"Steps loaded before calling `stop`: {self._steps_load_status}"
        )
        self._logger.info(
            "🛑 Stopping pipeline. Waiting for steps to finish processing batches..."
        )

        self._stop_load_queue_loop()
        self._stop_output_queue_loop()


class _ProcessWrapperException(Exception):
    """Exception to be raised when an error occurs in the `Step` process.

    Attributes:
        message: The error message.
        step: The `Step` that raised the error.
        code: The error code.
        subprocess_exception: The exception raised by the subprocess. Defaults to `None`.
    """

    def __init__(
        self,
        message: str,
        step: "_Step",
        code: int,
        subprocess_exception: Optional[Exception] = None,
    ) -> None:
        self.message = message
        self.step = step
        self.code = code
        self.subprocess_exception = subprocess_exception
        self.formatted_traceback = "".join(
            traceback.format_exception(subprocess_exception)
        )

    @classmethod
    def create_load_error(
        cls,
        message: str,
        step: "_Step",
        subprocess_exception: Optional[Exception] = None,
    ) -> "_ProcessWrapperException":
        """Creates a `_ProcessWrapperException` for a load error.

        Args:
            message: The error message.
            step: The `Step` that raised the error.
            subprocess_exception: The exception raised by the subprocess. Defaults to `None`.

        Returns:
            The `_ProcessWrapperException` instance.
        """
        return cls(message, step, 1, subprocess_exception)

    @property
    def is_load_error(self) -> bool:
        """Whether the error is a load error.

        Returns:
            `True` if the error is a load error, `False` otherwise.
        """
        return self.code == 1


class _ProcessWrapper:
    """Wrapper to run the `Step` in a separate process.

    Attributes:
        step: The step to run.
        replica: The replica ID assigned.
        input_queue: The queue to receive the input data.
        output_queue: The queue to send the output data.
        load_queue: The queue used to notify the main process that the step has been loaded,
            has been unloaded or has failed to load.
    """

    def __init__(
        self,
        step: Union["Step", "GeneratorStep"],
        replica: int,
        input_queue: "Queue[_Batch]",
        output_queue: "Queue[_Batch]",
        load_queue: "Queue[Union[StepLoadStatus, None]]",
        dry_run: bool = False,
    ) -> None:
        """Initializes the `_ProcessWrapper`.

        Args:
            step: The step to run.
            input_queue: The queue to receive the input data.
            output_queue: The queue to send the output data.
            load_queue: The queue used to notify the main process that the step has been
                loaded, has been unloaded or has failed to load.
            dry_run: Flag to ensure we are forcing to run the last batch.
        """
        self.step = step
        self.replica = replica
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.load_queue = load_queue
        self._dry_run = dry_run

        if (
            isinstance(self.step, Task)
            and hasattr(self.step, "llm")
            and isinstance(self.step.llm, CudaDevicePlacementMixin)
        ):
            self.step.llm._llm_identifier = self.step.name

    def run(self) -> str:
        """The target function executed by the process. This function will also handle
        the step lifecycle, executing first the `load` function of the `Step` and then
        waiting to receive a batch from the `input_queue` that will be handled by the
        `process` method of the `Step`.

        Returns:
            The name of the step that was executed.
        """

        try:
            self.step.load()
            self.step._logger.debug(f"Step '{self.step.name}' loaded!")
        except Exception as e:
            self.step.unload()
            self._notify_load_failed()
            raise _ProcessWrapperException.create_load_error(
                message=f"Step load failed: {e}",
                step=self.step,
                subprocess_exception=e,
            ) from e

        self._notify_load()

        if self.step.is_generator:
            self._generator_step_process_loop()
        else:
            self._non_generator_process_loop()

        # Just in case `None` sentinel was sent
        try:
            self.input_queue.get(block=False)
        except Exception:
            pass

        self.step.unload()

        self._notify_unload()

        self.step._logger.info(
            f"🏁 Finished running step '{self.step.name}' (replica ID: {self.replica})"
        )

        return self.step.name  # type: ignore

    def _notify_load(self) -> None:
        """Notifies that the step has finished executing its `load` function successfully."""
        self.load_queue.put({"name": self.step.name, "status": "loaded"})  # type: ignore

    def _notify_unload(self) -> None:
        """Notifies that the step has been unloaded."""
        self.load_queue.put({"name": self.step.name, "status": "unloaded"})  # type: ignore

    def _notify_load_failed(self) -> None:
        """Notifies that the step failed to load."""
        self.load_queue.put({"name": self.step.name, "status": "load_failed"})  # type: ignore

    def _generator_step_process_loop(self) -> None:
        """Runs the process loop for a generator step. It will call the `process` method
        of the step and send the output data to the `output_queue` and block until the next
        batch request is received (i.e. receiving an empty batch from the `input_queue`).

        If the `last_batch` attribute of the batch is `True`, the loop will stop and the
        process will finish.

        Raises:
            _ProcessWrapperException: If an error occurs during the execution of the
                `process` method.
        """
        step = cast("GeneratorStep", self.step)
        try:
            if (batch := self.input_queue.get()) is None:
                self.step._logger.info(
                    f"🛑 Stopping yielding batches from step '{self.step.name}'"
                )
                return

            offset = batch.seq_no * step.batch_size  # type: ignore

            self.step._logger.info(
                f"🧬 Starting yielding batches from generator step '{self.step.name}'."
                f" Offset: {offset}"
            )

            for data, last_batch in step.process_applying_mappings(offset=offset):
                batch.set_data([data])
                batch.last_batch = self._dry_run or last_batch
                self._send_batch(batch)

                if batch.last_batch:
                    return

                self.step._logger.debug(
                    f"Step '{self.step.name}' waiting for next batch request..."
                )
                if (batch := self.input_queue.get()) is None:
                    self.step._logger.info(
                        f"🛑 Stopping yielding batches from step '{self.step.name}'"
                    )
                    return
        except Exception as e:
            raise _ProcessWrapperException(str(e), self.step, 2, e) from e

    def _non_generator_process_loop(self) -> None:
        """Runs the process loop for a non-generator step. It will call the `process`
        method of the step and send the output data to the `output_queue` and block until
        the next batch is received from the `input_queue`. If the `last_batch` attribute
        of the batch is `True`, the loop will stop and the process will finish.

        If an error occurs during the execution of the `process` method and the step is
        global, the process will raise a `_ProcessWrapperException`. If the step is not
        global, the process will log the error and send an empty batch to the `output_queue`.

        Raises:
            _ProcessWrapperException: If an error occurs during the execution of the
                `process` method and the step is global.
        """
        step = cast("Step", self.step)
        while True:
            if (batch := self.input_queue.get()) is None:
                self.step._logger.info(
                    f"🛑 Stopping processing batches from step '{self.step.name}'"
                )
                break

            if batch == LAST_BATCH_SENT_FLAG:
                self.step._logger.debug("Received `LAST_BATCH_SENT_FLAG`. Stopping...")
                break

            self.step._logger.info(
                f"📦 Processing batch {batch.seq_no} in '{batch.step_name}' (replica ID: {self.replica})"
            )

            if batch.data_path is not None:
                self.step._logger.debug(f"Reading batch data from '{batch.data_path}'")
                batch.read_batch_data_from_fs()

            result = []
            try:
                if self.step.has_multiple_inputs:
                    result = next(step.process_applying_mappings(*batch.data))
                else:
                    result = next(step.process_applying_mappings(batch.data[0]))
            except Exception as e:
                if self.step.is_global:
                    raise _ProcessWrapperException(str(e), self.step, 2, e) from e

                # Impute step outputs columns with `None`
                result = self._impute_step_outputs(batch)

                # if the step is not global then we can skip the batch which means sending
                # an empty batch to the output queue
                self.step._logger.warning(
                    f"⚠️ Processing batch {batch.seq_no} with step '{self.step.name}' failed."
                    " Sending empty batch filled with `None`s..."
                )
                self.step._logger.warning(
                    f"Subprocess traceback:\n\n{traceback.format_exc()}"
                )
            finally:
                batch.set_data([result])
                self._send_batch(batch)

            if batch.last_batch:
                break

    def _impute_step_outputs(self, batch: "_Batch") -> List[Dict[str, Any]]:
        """Imputes the step outputs columns with `None` in the batch data.

        Args:
            batch: The batch to impute.
        """
        result = []
        for row in batch.data[0]:
            data = row.copy()
            for output in self.step.outputs:
                data[output] = None
            result.append(data)
        return result

    def _send_batch(self, batch: _Batch) -> None:
        """Sends a batch to the `output_queue`."""
        if batch.data_path is not None:
            self.step._logger.debug(f"Writing batch data to '{batch.data_path}'")
            batch.write_batch_data_to_fs()

        self.step._logger.info(
            f"📨 Step '{batch.step_name}' sending batch {batch.seq_no} to output queue"
        )
        self.output_queue.put(batch)
