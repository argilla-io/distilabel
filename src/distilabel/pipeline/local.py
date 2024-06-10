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
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import tblib

from distilabel.distiset import create_distiset
from distilabel.llms.mixins import CudaDevicePlacementMixin
from distilabel.pipeline.base import (
    BasePipeline,
)
from distilabel.pipeline.batch import _Batch
from distilabel.pipeline.constants import (
    INPUT_QUEUE_ATTR_NAME,
    LAST_BATCH_SENT_FLAG,
    STEP_ATTR_NAME,
)
from distilabel.steps.base import Step
from distilabel.utils.logging import setup_logging, stop_logging

if TYPE_CHECKING:
    from multiprocessing.managers import DictProxy, SyncManager
    from multiprocessing.pool import Pool
    from queue import Queue

    from distilabel.distiset import Distiset
    from distilabel.steps.base import GeneratorStep


_STEPS_LOADED_KEY = "steps_loaded"
_STEPS_LOADED_LOCK_KEY = "steps_loaded_lock"
_STEPS_LOADED_ERROR_CODE = -1
_CUDA_LLM_DEVICE_PLACEMENT_KEY = "cuda_llm_device_placement"
_CUDA_LLM_DEVICE_PLACEMENT_LOCK_KEY = "cuda_llm_device_placement_lock"

_STOP_CALLED = False
_STOP_CALLED_LOCK = threading.Lock()
_STOP_CALLS = 0

_STEPS_LOADED = set()
_STEPS_LOADED_LOCK = threading.Lock()

_STEPS_FINISHED = set()
_STEPS_FINISHED_LOCK = threading.Lock()

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

        num_processes = len(self.dag)
        ctx = mp.get_context()  # type: ignore
        with ctx.Manager() as manager, ctx.Pool(
            num_processes,
            initializer=_init_worker,
            initargs=(log_queue,),
        ) as pool:
            self.output_queue: "Queue[Any]" = manager.Queue()
            self.shared_info = self._create_shared_info_dict(manager)
            self._handle_keyboard_interrupt(manager=manager, pool=pool)

            # Run the steps using the pool of processes
            self._run_steps_in_loop(pool, manager, self.output_queue, self.shared_info)

            # Wait for all the steps to be loaded correctly
            if not self._all_steps_loaded():
                self._write_buffer.close()  # type: ignore
                self._batch_manager = None
                stop_logging()
                raise RuntimeError(
                    "Failed to load all the steps. Could not run pipeline."
                ) from _SUBPROCESS_EXCEPTION

            # Send the "first" batches to the steps so the batches starts flowing through
            # the input queues and output queue
            self._request_initial_batches()

            # Start a loop to receive the output batches from the steps
            self._run_output_queue_loop_in_thread()

            # Send `None` to steps `input_queue`s just in case some step is still waiting
            self._notify_steps_to_stop()

        # `Pool.__exit__` has already called `terminate`, `join` the pool to make sure
        # all the processes have finished
        pool.join()
        manager.join()

        self._write_buffer.close()  # type: ignore
        distiset = create_distiset(
            self._cache_location["data"],
            pipeline_path=self._cache_location["pipeline"],
            log_filename_path=self._cache_location["log_file"],
            enable_metadata=self._enable_metadata,
        )
        stop_logging()
        return distiset

    def _run_output_queue_loop_in_thread(self) -> None:
        """Runs the output queue loop in a separate thread to receive the output batches
        from the steps. This is done to avoid the signal handler to block the loop, which
        would prevent the pipeline from stopping correctly."""
        thread = threading.Thread(target=self._output_queue_loop)
        thread.start()
        thread.join()

    def _output_queue_loop(self) -> None:
        """Loop to receive the output batches from the steps and manage the flow of the
        batches through the pipeline."""
        while self._batch_manager.can_generate() and not _STOP_CALLED:  # type: ignore
            self._logger.debug("Waiting for output batch from step...")
            if (batch := self._get_from_step()) is None:
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

            # If `_STOP_CALLED` was set to `True` while waiting for the output queue, then
            # we need to handle the stop of the pipeline and break the loop to avoid
            # propagating the batches through the pipeline and making the stop process
            # slower.
            if _STOP_CALLED:
                self._handle_batch_on_stop(batch)
                break

            self._manage_batch_flow(batch)

        if _STOP_CALLED:
            self._handle_stop()

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

    def _create_shared_info_dict(self, manager: "SyncManager") -> "DictProxy[str, Any]":
        """Creates the shared information dictionary to be used by the processes.

        Args:
            manager: The manager to create the shared information.

        Returns:
            The shared information dictionary.
        """
        # TODO: not very important, but we could use a different lock for each matter
        return manager.dict(
            **{
                _STEPS_LOADED_KEY: manager.list(),
                _STEPS_LOADED_LOCK_KEY: manager.Lock(),
                _CUDA_LLM_DEVICE_PLACEMENT_KEY: manager.dict(**{}),
                _CUDA_LLM_DEVICE_PLACEMENT_LOCK_KEY: manager.Lock(),
            }
        )

    def _all_steps_loaded(self) -> bool:
        """Waits for all the steps to load.

        Returns:
            `True` if all the steps have been loaded correctly, `False` otherwise.
        """

        def _update_all_steps_loaded(steps_loaded: List[str]) -> None:
            with _STEPS_LOADED_LOCK:
                _STEPS_LOADED.update(steps_loaded)

        self._logger.info("⏳ Waiting for all the steps to load...")
        previous_message = None
        while not _STOP_CALLED:
            with self.shared_info[_STEPS_LOADED_LOCK_KEY]:
                steps_loaded = self.shared_info[_STEPS_LOADED_KEY]
                num_steps_loaded = (
                    len(steps_loaded)
                    if steps_loaded != [_STEPS_LOADED_ERROR_CODE]
                    else 0
                )
                self._logger.debug(f"Steps loaded: {steps_loaded}")

                message = f"⏳ Steps loaded: {num_steps_loaded}/{len(self.dag)}"
                if num_steps_loaded > 0 and message != previous_message:
                    self._logger.info(message)
                    previous_message = message

                if num_steps_loaded == len(self.dag):
                    self._logger.info("✅ All the steps have been loaded!")
                    _update_all_steps_loaded(steps_loaded)
                    return True

                if steps_loaded == [_STEPS_LOADED_ERROR_CODE]:
                    self._logger.error("❌ Failed to load all the steps")
                    _update_all_steps_loaded(steps_loaded)
                    return False

            time.sleep(2.5)

        return not _STOP_CALLED

    def _run_steps_in_loop(
        self,
        pool: "Pool",
        manager: "SyncManager",
        output_queue: "Queue[_Batch]",
        shared_info: "DictProxy[str, Any]",
    ) -> None:
        """Using the `pool`, runs the steps in the DAG in an infinite loop waiting for
        input batches and sending the output batches to the `output_queue`.

        Each `Step` is wrapped in a `_ProcessWrapper`, which will handle the lifecycle of
        the `Step` and the communication with the `input_queue` and `output_queue`. The
        `_ProcessWrapper.run` method is the target function of the process.

        Args:
            pool: The pool of processes.
            manager: The manager to create the queues.
            output_queue: The queue to send the output batches.
            shared_info: The shared information between the processes.
        """
        for step_name in self.dag:
            step: "Step" = self.dag.get_step(step_name)[STEP_ATTR_NAME]
            input_queue = manager.Queue()
            self.dag.set_step_attr(step.name, INPUT_QUEUE_ATTR_NAME, input_queue)  # type: ignore

            # Set `pipeline` to `None` as in some Python environments the pipeline is not
            # picklable and it will raise an error when trying to send the step to the process.
            # `TypeError: cannot pickle 'code' object`
            step.pipeline = None

            process_wrapper = _ProcessWrapper(
                step=step,
                input_queue=input_queue,
                output_queue=output_queue,
                shared_info=shared_info,
                dry_run=self._dry_run,
            )

            pool.apply_async(
                process_wrapper.run,
                callback=self._finished_callback,
                error_callback=self._error_callback,
            )  # type: ignore

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
            with self.shared_info[_STEPS_LOADED_LOCK_KEY]:
                self.shared_info[_STEPS_LOADED_KEY] = [_STEPS_LOADED_ERROR_CODE]
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
        self._cache()
        self._stop()

    def _finished_callback(self, step_name: str) -> None:
        """Callback that will be called when a `Step` process finishes.

        Args:
            step_name: The name of the step that finished.
        """
        with _STEPS_FINISHED_LOCK:
            _STEPS_FINISHED.add(step_name)

    def _check_step_not_loaded_or_finished(self, step_name: str) -> bool:
        """Checks if a step is not loaded or already finished.

        Args:
            step_name: The name of the step.

        Returns:
            `True` if the step is not loaded or already finished, `False` otherwise.
        """
        with _STEPS_LOADED_LOCK:
            if step_name not in _STEPS_LOADED:
                return True

        with _STEPS_FINISHED_LOCK:
            if step_name in _STEPS_FINISHED:
                return True

        return False

    def _stop(
        self, manager: Optional["SyncManager"] = None, pool: Optional["Pool"] = None
    ) -> None:
        """Stops the pipeline execution. It will first send `None` to the input queues
        of all the steps and then wait until the output queue is empty i.e. all the steps
        finished processing the batches that were sent before the stop flag. Then it will
        send `None` to the output queue to notify the pipeline to stop."""

        global _STOP_CALLED

        with _STOP_CALLED_LOCK:
            if _STOP_CALLED:
                global _STOP_CALLS
                _STOP_CALLS += 1
                if _STOP_CALLS == 1:
                    self._logger.warning(
                        "🛑 Press again to force the pipeline to stop."
                    )
                elif _STOP_CALLS > 1:
                    self._logger.warning("🛑 Forcing pipeline interruption.")

                    if pool:
                        pool.terminate()
                        pool.join()

                    if manager:
                        manager.shutdown()
                        manager.join()

                    stop_logging()

                    sys.exit(1)

                return
            _STOP_CALLED = True

        self._logger.debug(f"Steps loaded before calling `stop`: {_STEPS_LOADED}")
        self._logger.info(
            "🛑 Stopping pipeline. Waiting for steps to finish processing batches..."
        )
        self._logger.debug("Sending `None` to the output queue to notify stop...")
        self.output_queue.put(None)

    def _handle_keyboard_interrupt(
        self, manager: Optional["SyncManager"] = None, pool: Optional["Pool"] = None
    ) -> None:
        """Handles KeyboardInterrupt signal sent during the Pipeline.run method.

        It will try to call self._stop (if the pipeline didn't started yet, it won't
        have any effect), and if the pool is already started, will close it before exiting
        the program.
        """

        def signal_handler(signumber: int, frame: Any) -> None:
            self._stop(manager=manager, pool=pool)

        signal.signal(signal.SIGINT, signal_handler)


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
        step: "Step",
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
        step: "Step",
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
        input_queue: The queue to receive the input data.
        output_queue: The queue to send the output data.
        shared_info: The shared information between the processes.
    """

    def __init__(
        self,
        step: "Step",
        input_queue: "Queue[_Batch]",
        output_queue: "Queue[_Batch]",
        shared_info: "DictProxy[str, Any]",
        dry_run: bool = False,
    ) -> None:
        """Initializes the `_ProcessWrapper`.

        Args:
            step: The step to run.
            input_queue: The queue to receive the input data.
            output_queue: The queue to send the output data.
            shared_info: The shared information between the processes.
            dry_run: Flag to ensure we are forcing to run the last batch.
        """
        self.step = step
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shared_info = shared_info
        self._dry_run = dry_run

        # If step is a task, and it's using a `CUDALLM`, then set the CUDA device map
        # and the lock for that map.
        if hasattr(self.step, "llm") and isinstance(
            self.step.llm, CudaDevicePlacementMixin
        ):
            self.step.llm.set_device_placement_info(
                llm_identifier=self.step.name,
                device_llm_placement_map=self.shared_info[
                    _CUDA_LLM_DEVICE_PLACEMENT_KEY
                ],
                device_llm_placement_lock=self.shared_info[
                    _CUDA_LLM_DEVICE_PLACEMENT_LOCK_KEY
                ],
            )

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
            raise _ProcessWrapperException.create_load_error(
                str(e), self.step, e
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

        self.step._logger.info(f"🏁 Finished running step '{self.step.name}'")

        return self.step.name  # type: ignore

    def _notify_load(self) -> None:
        """Notifies that the step has finished executing its `load` function successfully."""
        with self.shared_info[_STEPS_LOADED_LOCK_KEY]:
            self.shared_info[_STEPS_LOADED_KEY].append(self.step.name)

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
                f"📦 Processing batch {batch.seq_no} in '{batch.step_name}'"
            )

            if batch.data_path is not None:
                self.step._logger.debug(f"Reading batch data from '{batch.data_path}'")
                batch.read_batch_data_from_fs()

            result = []
            try:
                if self.step.has_multiple_inputs:
                    result = next(self.step.process_applying_mappings(*batch.data))
                else:
                    result = next(self.step.process_applying_mappings(batch.data[0]))
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
