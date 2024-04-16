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

import logging
import multiprocessing as mp
import signal
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import tblib

from distilabel.distiset import create_distiset
from distilabel.llms.mixins import CudaDevicePlacementMixin
from distilabel.pipeline.base import BasePipeline, _Batch, _BatchManager, _WriteBuffer
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

_STEPS_LOADED = set()
_STEPS_LOADED_LOCK = threading.Lock()

_STEPS_FINISHED = set()
_STEPS_FINISHED_LOCK = threading.Lock()

_SUBPROCESS_EXCEPTION: Union[Exception, None] = None


def _init_worker(queue: "Queue[Any]") -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    setup_logging(queue)


class Pipeline(BasePipeline):
    """Local pipeline implementation using `multiprocessing`."""

    def run(
        self,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache: bool = True,
    ) -> "Distiset":
        """Runs the pipeline.

        Args:
            parameters: A dictionary with the step name as the key and a dictionary with
                the runtime parameters for the step as the value. Defaults to `None`.
            use_cache: Whether to use the cache from previous pipeline runs. Defaults to
                `True`.

        Returns:
            The `Distiset` created by the pipeline.

        Raises:
            RuntimeError: If the pipeline fails to load all the steps.
        """
        try:
            mp.set_start_method("forkserver")
        except RuntimeError:
            pass
        log_queue = mp.Queue()
        setup_logging(log_queue)  # type: ignore
        self._logger = logging.getLogger("distilabel.pipeline.local")

        super().run(parameters, use_cache)

        if self._batch_manager is None:
            self._batch_manager = _BatchManager.from_dag(self.dag)

        # If the batch manager is not able to generate batches, that means that the loaded
        # `_BatchManager` from cache didn't have any remaining batches to process i.e.
        # the previous pipeline execution was completed successfully.
        if not self._batch_manager.can_generate():
            self._logger.info(
                "💾 Loaded batch manager from cache doesn't have any remaining data. Returning"
                " `Distiset` from cache data..."
            )
            stop_logging()
            return create_distiset(
                self._cache_location["data"],
                pipeline_path=self._cache_location["pipeline"],
            )

        buffer_data_path = self._cache_location["data"]
        self._logger.info(f"📝 Pipeline data will be written to '{buffer_data_path}'")
        write_buffer = _WriteBuffer(buffer_data_path, self.dag.leaf_steps)

        num_processes = len(self.dag)
        ctx = mp.get_context("forkserver")  # type: ignore
        with ctx.Manager() as manager, ctx.Pool(
            num_processes, initializer=_init_worker, initargs=(log_queue,)
        ) as pool:
            self.output_queue: "Queue[Any]" = manager.Queue()
            self.shared_info = self._create_shared_info_dict(manager)
            self._handle_keyboard_interrupt()

            # Run the steps using the pool of processes
            self._run_steps_in_loop(pool, manager, self.output_queue, self.shared_info)

            # Wait for all the steps to be loaded correctly
            if not self._all_steps_loaded():
                write_buffer.close()
                self._batch_manager = None
                stop_logging()
                raise RuntimeError(
                    "Failed to load all the steps. Could not run pipeline."
                ) from _SUBPROCESS_EXCEPTION

            # Send the "first" batches to the steps so the batches starts flowing through
            # the input queues and output queue
            self._request_initial_batches()

            # Start a loop to receive the output batches from the steps
            self._run_output_queue_loop_in_thread(write_buffer)

            # Send `None` to steps `input_queue`s just in case some step is still waiting
            self._notify_steps_to_stop()

            pool.close()
            pool.join()

        write_buffer.close()
        distiset = create_distiset(
            self._cache_location["data"], pipeline_path=self._cache_location["pipeline"]
        )
        stop_logging()
        return distiset

    def _run_output_queue_loop_in_thread(self, write_buffer: "_WriteBuffer") -> None:
        """Runs the output queue loop in a separate thread to receive the output batches
        from the steps. This is done to avoid the signal handler to block the loop, which
        would prevent the pipeline from stopping correctly.

        Args:
            write_buffer: The write buffer to write the data from the leaf steps to disk.
        """
        thread = threading.Thread(target=self._output_queue_loop, args=(write_buffer,))
        thread.start()
        thread.join()

    def _notify_steps_to_stop(self) -> None:
        """Notifies the steps to stop their infinite running loop by sending `None` to
        their input queues."""
        for step_name in self.dag:
            if input_queue := self.dag.get_step(step_name).get("input_queue"):
                input_queue.put(None)

    def _output_queue_loop(self, write_buffer: "_WriteBuffer") -> None:
        """Loop to receive the output batches from the steps and manage the flow of the
        batches through the pipeline.

        Args:
            write_buffer: The write buffer to write the data from the leaf steps to disk.
        """
        while self._batch_manager.can_generate() and not _STOP_CALLED:  # type: ignore
            self._logger.debug("Waiting for output batch from step...")
            if (batch := self.output_queue.get()) is None:
                self._logger.debug("Received `None` from output queue. Breaking loop.")
                break

            if batch.step_name in self.dag.leaf_steps:
                write_buffer.add_batch(batch)

            # If `_STOP_CALLED` was set to `True` while waiting for the output queue, then
            # we need to handle the stop of the pipeline and break the loop to avoid
            # propagating the batches through the pipeline and making the stop process
            # slower.
            if _STOP_CALLED:
                self._handle_batch_on_stop(batch)
                break

            self._logger.debug(
                f"Received batch with seq_no {batch.seq_no} from step '{batch.step_name}'"
                f" from output queue: {batch}"
            )

            self._manage_batch_flow(batch)

        if _STOP_CALLED:
            self._handle_stop(write_buffer)

    def _manage_batch_flow(self, batch: "_Batch") -> None:
        """Checks if the step that generated the batch has more data in its buffer to
        generate a new batch. If there's data, then a new batch is sent to the step. If
        the step has no data in its buffer, then the predecessors generator steps are
        requested to send a new batch.

        Args:
            batch: The batch that was processed.
        """
        assert self._batch_manager, "Batch manager is not set"

        self._batch_manager.register_batch(batch)
        self._logger.debug(
            f"Batch {batch.seq_no} from step '{batch.step_name}' registered in batch"
            " manager"
        )

        step: "Step" = self.dag.get_step(batch.step_name)["step"]

        for successor in self.dag.get_step_successors(step.name):
            self._batch_manager.add_batch(successor, batch)

            # Check if the step is a generator and if there are successors that need data
            # from this step. This usually happens when the generator `batch_size` is smaller
            # than the `input_batch_size` of the successor steps.
            if (
                step.is_generator
                and step.name in self._batch_manager.step_empty_buffers(successor)
            ):
                last_batch = self._batch_manager.get_last_batch(step.name)
                self._send_batch_to_step(last_batch.next_batch())  # type: ignore

            if new_batch := self._batch_manager.get_batch(successor):
                self._send_batch_to_step(new_batch)

        if step.is_generator:
            return

        # Step has enough data on its buffers to create a new batch
        if next_batch := self._batch_manager.get_batch(step.name):
            self._send_batch_to_step(next_batch)
            return

        # Request more batches to the predecessors generator steps
        empty_buffers = self._batch_manager.step_empty_buffers(step.name)
        for previous_step_name in empty_buffers:
            if previous_step_name not in self.dag.root_steps:
                continue

            if last_batch := self._batch_manager.get_last_batch(previous_step_name):
                self._logger.debug(
                    f"Step '{step.name}' input buffer for step '{previous_step_name}' is"
                    " empty. Requesting new batch..."
                )
                self._send_batch_to_step(last_batch.next_batch())

        self._cache()

    def _handle_stop(self, write_buffer: "_WriteBuffer") -> None:
        """Handles the stop of the pipeline execution, which will stop the steps from
        processing more batches and wait for the output queue to be empty, to not lose
        any data that was already processed by the steps before the stop was called.

        Args:
            write_buffer: The write buffer to write the data from the leaf steps to disk.
        """
        self._logger.debug("Handling stop of the pipeline execution...")

        # Send `None` to the input queues of all the steps to notify them to stop
        # processing batches.
        for step_name in self.dag:
            if input_queue := self.dag.get_step(step_name).get("input_queue"):
                while not input_queue.empty():
                    batch = input_queue.get()
                    self._batch_manager.add_batch(  # type: ignore
                        to_step=step_name, batch=batch, prepend=True
                    )
                    self._logger.debug(
                        f"Adding batch back to the batch manager: {batch}"
                    )
                input_queue.put(None)

        # Wait for the input queue to be empty, which means that all the steps finished
        # processing the batches that were sent before the stop flag.
        for step_name in self.dag:
            self._wait_step_input_queue_empty(step_name)

        # Consume the output queue until it's empty to not lose any data that was already
        # processed by the steps before stop was called.
        while not self.output_queue.empty():
            batch = self.output_queue.get()
            if batch.step_name in self.dag.leaf_steps:
                write_buffer.add_batch(batch)
            self._handle_batch_on_stop(batch)

        self._cache()

    def _handle_batch_on_stop(self, batch: "_Batch") -> None:
        """Handles a batch that was received from the output queue when the pipeline was
        stopped. It will add and register the batch in the batch manager.

        Args:
            batch: The batch to handle.
        """
        self._batch_manager.register_batch(batch)  # type: ignore
        step: "Step" = self.dag.get_step(batch.step_name)["step"]
        for successor in self.dag.get_step_successors(step.name):
            self._batch_manager.add_batch(successor, batch)  # type: ignore

    def _wait_step_input_queue_empty(self, step_name: str) -> Union["Queue[Any]", None]:
        """Waits for the input queue of a step to be empty.

        Args:
            step_name: The name of the step.

        Returns:
            The input queue of the step if it's not loaded or finished, `None` otherwise.
        """
        if self._check_step_not_loaded_or_finished(step_name):
            return None

        if input_queue := self.dag.get_step(step_name).get("input_queue"):
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
            batch = _Batch(seq_no=seq_no, step_name=step_name, last_batch=False)
            self._logger.debug(
                f"Requesting initial batch to '{step_name}' generator step: {batch}"
            )
            self._send_batch_to_step(batch)

    def _send_batch_to_step(self, batch: "_Batch") -> None:
        """Sends a batch to the input queue of a step.

        Args:
            batch: The batch to send.
        """
        self._logger.debug(
            f"Sending batch {batch.seq_no} to step '{batch.step_name}': {batch}"
        )
        input_queue = self.dag.get_step(batch.step_name)["input_queue"]
        input_queue.put(batch)

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
            step: "Step" = self.dag.get_step(step_name)["step"]
            input_queue = manager.Queue()
            self.dag.set_step_attr(step.name, "input_queue", input_queue)

            # Set `pipeline` to `None` as in some Python environments the pipeline is not
            # picklable and it will raise an error when trying to send the step to the process.
            # `TypeError: cannot pickle 'code' object`
            step.pipeline = None

            process_wrapper = _ProcessWrapper(
                step=step,
                input_queue=input_queue,
                output_queue=output_queue,
                shared_info=shared_info,
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
            _SUBPROCESS_EXCEPTION.__traceback__ = tblib.Traceback.from_string(
                e.formatted_traceback
            ).as_traceback()
            return

        # If the step is global, is not in the last trophic level and has no successors,
        # then we can ignore the error and continue executing the pipeline
        if (
            e.step.is_global
            and not self.dag.step_in_last_trophic_level(e.step.name)
            and list(self.dag.get_step_successors(e.step.name)) == []
        ):
            self._logger.error(
                f"✋ An error occurred when running global step '{e.step.name}' with no"
                " successors and not in the last trophic level. Pipeline execution can"
                f" continue. Error will be ignored."
            )
            self._logger.error(f"Subprocess traceback:\n\n{e.formatted_traceback}")
            return

        # Global step with successors failed
        self._logger.error(f"An error occurred in global step '{e.step.name}'")
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

    def _stop(self) -> None:
        """Stops the pipeline execution. It will first send `None` to the input queues
        of all the steps and then wait until the output queue is empty i.e. all the steps
        finished processing the batches that were sent before the stop flag. Then it will
        send `None` to the output queue to notify the pipeline to stop."""

        global _STOP_CALLED

        with _STOP_CALLED_LOCK:
            if _STOP_CALLED:
                self._logger.warning(
                    "🛑 Stop has already been called. Ignoring subsequent calls and waiting"
                    " for the pipeline to finish..."
                )
                return
            _STOP_CALLED = True

        self._logger.debug(f"Steps loaded before calling `stop`: {_STEPS_LOADED}")
        self._logger.info(
            "🛑 Stopping pipeline. Waiting for steps to finish processing batches..."
        )
        self._logger.debug("Sending `None` to the output queue to notify stop...")
        self.output_queue.put(None)

    def _handle_keyboard_interrupt(self) -> None:
        """Handles KeyboardInterrupt signal sent during the Pipeline.run method.

        It will try to call self._stop (if the pipeline didn't started yet, it won't
        have any effect), and if the pool is already started, will close it before exiting
        the program.
        """

        def signal_handler(signumber: int, frame: Any) -> None:
            self._stop()

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
    ) -> None:
        """Initializes the `_ProcessWrapper`.

        Args:
            step: The step to run.
            input_queue: The queue to receive the input data.
            output_queue: The queue to send the output data.
            shared_info: The shared information between the processes.
        """
        self.step = step
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shared_info = shared_info

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

        return self.step.name

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

            offset = batch.seq_no * step.batch_size

            self.step._logger.info(
                f"🧬 Starting yielding batches from generator step '{self.step.name}'."
                f" Offset: {offset}"
            )

            for data, last_batch in step.process_applying_mappings(offset=offset):
                batch.data = [data]
                batch.last_batch = last_batch
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

            self.step._logger.info(
                f"📦 Processing batch {batch.seq_no} in '{batch.step_name}'"
            )
            # `result` is initially an empty list so f `process` method raises an exception
            # an empty batch will be sent to the `output_queue`
            result = []
            try:
                if self.step.has_multiple_inputs:
                    result = next(self.step.process_applying_mappings(*batch.data))
                else:
                    result = next(self.step.process_applying_mappings(batch.data[0]))
            except Exception as e:
                if self.step.is_global:
                    raise _ProcessWrapperException(str(e), self.step, 2, e) from e

                # if the step is not global then we can skip the batch which means sending
                # an empty batch to the output queue
                self.step._logger.warning(
                    f"⚠️ Processing batch {batch.seq_no} with step '{self.step.name}' failed."
                    " Sending empty batch..."
                )
                self.step._logger.warning(
                    f"Subprocess traceback:\n\n{traceback.format_exc()}"
                )
            finally:
                batch.data = [result]
                self._send_batch(batch)

            if batch.last_batch:
                break

    def _send_batch(self, batch: _Batch) -> None:
        """Sends a batch to the `output_queue`."""
        self.step._logger.info(
            f"📨 Step '{batch.step_name}' sending batch {batch.seq_no} to output queue"
        )
        self.output_queue.put(batch)
