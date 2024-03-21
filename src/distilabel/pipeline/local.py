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
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from distilabel.llm.mixins import CudaDevicePlacementMixin
from distilabel.pipeline.base import BasePipeline, _Batch, _BatchManager, _WriteBuffer
from distilabel.steps.base import Step
from distilabel.steps.task.base import _Task
from distilabel.utils.distiset import _create_dataset

if TYPE_CHECKING:
    from multiprocessing.managers import DictProxy, SyncManager
    from multiprocessing.pool import Pool
    from queue import Queue

    from distilabel.steps.base import GeneratorStep
    from distilabel.utils.distiset import Distiset

_BATCH_STOP_FLAG = "__STOP__"

_STEPS_LOADED_KEY = "steps_loaded"
_STEPS_LOADED_LOCK = "steps_loaded_lock"
_STEPS_LOADED_ERROR_CODE = -1
_CUDA_LLM_DEVICE_PLACEMENT_KEY = "cuda_llm_device_placement"
_CUDA_LLM_DEVICE_PLACEMENT_LOCK = "cuda_llm_device_placement_lock"

_STOP_CALLED = False
_STOP_CALLED_LOCK = threading.Lock()


class Pipeline(BasePipeline):
    """Local pipeline implementation using `multiprocessing`."""

    def run(self, parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> "Distiset":
        """Runs the pipeline.

        Args:
            parameters: a dictionary containing the runtime parameters for each step.
                The keys are the step names and the values are dictionaries in which the
                keys are the parameter names (defined in the `process` method of the step)
                and the values are the parameter values.

        Returns:
            The `Distiset` created by the pipeline.

        Raises:
            RuntimeError: If the pipeline fails to load all the steps.
        """
        super().run(parameters)

        if self._batch_manager is None:
            self._batch_manager = _BatchManager.from_dag(self.dag)

        # If the batch manager is not able to generate batches, that means that the loaded
        # `_BatchManager` from cache didn't have any remaining batches to process i.e.
        # the previous pipeline execution was completed successfully.
        if not self._batch_manager.can_generate():
            self._logger.info(
                "Loaded batch manager from cache doesn't have any remaining data. Returning"
                " `Distiset` from cache data"
            )
            return _create_dataset(self._cache_location["data"])

        buffer_data_path = self._cache_location["data"]
        self._logger.info(f"📝 Pipeline data will be written to '{buffer_data_path}'")
        write_buffer = _WriteBuffer(buffer_data_path, self.dag.leaf_steps)

        num_processes = len(self.dag)
        ctx = mp.get_context("forkserver")
        with ctx.Manager() as manager, ctx.Pool(num_processes) as pool:
            self.output_queue: "Queue[Any]" = manager.Queue()
            self.shared_info = self._create_shared_info_dict(manager)
            self._handle_keyboard_interrupt()

            # Run the steps using the pool of processes
            self._run_steps_in_loop(pool, manager, self.output_queue, self.shared_info)

            # Wait for all the steps to be loaded correctly
            if not self._all_steps_loaded():
                write_buffer.close()
                raise RuntimeError(
                    "Failed to load all the steps. Could not run pipeline."
                )

            # Send the "first" batches to the steps so the batches starts flowing through
            # the input queues and output queue
            self._request_initial_batches()

            # Start a loop to receive the output batches from the steps
            self._output_queue_loop(write_buffer)

            pool.close()
            pool.join()

        write_buffer.close()
        return _create_dataset(self._cache_location["data"])

    def _output_queue_loop(self, write_buffer: "_WriteBuffer") -> None:
        """Loop to receive the output batches from the steps and manage the flow of the
        batches through the pipeline.

        Args:
            write_buffer: The write buffer to write the data from the leaf steps to disk.
        """
        while self._batch_manager.can_generate():  # type: ignore
            self._logger.debug("Waiting for output batch from step...")
            if (batch := self.output_queue.get()) == _BATCH_STOP_FLAG or batch is None:
                self._logger.debug(
                    "Received `_BATCH_STOP_FLAG` from output queue. Breaking loop."
                )
                break

            self._logger.debug(
                f"Received {batch.seq_no} from step '{batch.step_name}' from output"
                f" queue: {batch}"
            )

            self._add_batch_to_batch_manager(batch)

            self._manage_batch_flow(batch)

            if batch.step_name in self.dag.leaf_steps:
                write_buffer.add_batch(batch.step_name, batch)

    def _add_batch_to_batch_manager(self, batch: "_Batch") -> None:
        """Registers the batch in the `_BatchManager` and adds the batch to input buffer
        of the successors steps from the step that generated the batch. If there's enough
        data for creating a batch for a successor step, then the batch is created and sent
        to that step.

        Args:
            batch: The batch to add to the `_BatchManager`.
        """
        # Register the batch so we can keep track of the last batch processed
        # by each step
        self._batch_manager.register_batch(batch, lambda: self._cache())  # type: ignore

        # Add the received batch to the `_BatchManager`, and send a new batch
        # to successors if there is enough data
        for step_name in self.dag.get_step_successors(batch.step_name):
            if new_batch := self._batch_manager.add_batch(  # type: ignore
                to_step=step_name, batch=batch, callback=lambda: self._cache()
            ):
                self._send_batch_to_step(new_batch)

    def _manage_batch_flow(self, batch: "_Batch") -> None:
        """Checks if the step that generated the batch has more data in its buffer to
        generate a new batch. If there's data, then a new batch is sent to the step. If
        the step has no data in its buffer, then the predecessors generator steps are
        requested to send a new batch.

        Args:
            batch: The batch that was processed.
        """
        if batch.last_batch:
            return

        step: "Step" = self.dag.get_step(batch.step_name)["step"]
        if not step.is_normal:
            return

        empty_buffers = self._batch_manager.step_empty_buffers(step.name)  # type: ignore

        # Step has data in its buffers, send a new batch
        if not empty_buffers and (
            next_batch := self._batch_manager.get_batch(step.name)  # type: ignore
        ):
            self._send_batch_to_step(next_batch)
            return

        # Request more batches to the predecessors generator steps
        for step_name in empty_buffers:
            if step_name not in self.dag.root_steps:
                continue

            if previous_batch := self._batch_manager.get_last_batch(  # type: ignore
                step_name
            ):
                self._logger.debug(
                    f"Step '{step.name}' input buffer for step '{step_name}' is empty."
                    " Requesting new batch..."
                )
                self._send_batch_to_step(previous_batch.next_batch())

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
                _STEPS_LOADED_KEY: 0,
                _STEPS_LOADED_LOCK: manager.Lock(),
                _CUDA_LLM_DEVICE_PLACEMENT_KEY: manager.dict(**{}),
                _CUDA_LLM_DEVICE_PLACEMENT_LOCK: manager.Lock(),
            }
        )

    def _all_steps_loaded(self) -> bool:
        """Waits for all the steps to load.

        Returns:
            `True` if all the steps have been loaded correctly, `False` otherwise.
        """
        self._logger.info("⏳ Waiting for all the steps to load...")
        previous_message = None
        while True:
            with self.shared_info[_STEPS_LOADED_LOCK]:
                steps_loaded = self.shared_info[_STEPS_LOADED_KEY]

                message = f"⏳ Steps loaded: {steps_loaded}/{len(self.dag)}"
                if steps_loaded > 0 and message != previous_message:
                    self._logger.info(message)
                    previous_message = message

                if steps_loaded == len(self.dag):
                    self._logger.info("✅ All the steps have been loaded!")
                    return True

                if steps_loaded == _STEPS_LOADED_ERROR_CODE:
                    self._logger.error("❌ Failed to load all the steps")
                    return False

            time.sleep(2.5)

    def _request_initial_batches(self) -> None:
        """Requests the initial batches to the generator steps."""
        assert self._batch_manager, "Batch manager is not set"

        for step in self._batch_manager._steps.values():
            if batch := step.get_batch():
                self._send_batch_to_step(batch)

        for step_name in self.dag.root_steps:
            seq_no = 0
            if last_batch := self._batch_manager.get_last_batch(step_name):
                seq_no = last_batch.seq_no + 1
            batch = _Batch(seq_no=seq_no, step_name=step_name, last_batch=False)
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
            step = self.dag.get_step(step_name)["step"]
            input_queue = manager.Queue()
            self.dag.set_step_attr(step.name, "input_queue", input_queue)

            process_wrapper = _ProcessWrapper(
                step=step,
                input_queue=input_queue,
                output_queue=output_queue,
                shared_info=shared_info,
            )

            pool.apply_async(process_wrapper.run, error_callback=self._error_callback)  # type: ignore

    def _error_callback(self, e: Exception) -> None:
        """Error callback that will be called when an error occurs in a `Step` process.

        Args:
            e: The exception raised by the process.
        """
        # First we check that the exception is a `_ProcessWrapperException`, otherwise, we
        # print it out and stop the pipeline, since some errors may be unhandled
        if not isinstance(e, _ProcessWrapperException):
            self._logger.error(f"❌ Failed with an unhandled exception: {e}")
            self._stop()
            return

        if e.is_load_error:
            self._logger.error(f"❌ Failed to load step '{e.step.name}': {e.message}")
            self._stop()
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
                f" continue. Error will be ignored: {e.message}"
            )
            return

        self._logger.error(f"An error occurred in step '{e.step.name}': {e.message}")
        self._cache()
        self._stop()

    def _stop(self) -> None:
        """Stops the pipeline execution. It will first send the `_BATCH_STOP_FLAG` to the
        input queues of all the steps and then wait until the output queue is empty i.e.
        all the steps finished processing the batches that were sent before the stop flag.
        Then it will send the `_BATCH_STOP_FLAG` to the output queue to notify the pipeline
        to stop."""

        global _STOP_CALLED

        with _STOP_CALLED_LOCK:
            if _STOP_CALLED:
                return
            _STOP_CALLED = True

        for step_name in self.dag:
            if input_queue := self.dag.get_step(step_name).get("input_queue"):
                input_queue.put(_BATCH_STOP_FLAG)
                self._logger.debug(
                    f"Send `_BATCH_STOP_FLAG` to step '{step_name}' input queue."
                )
        # Wait until the output queue is empty which means that all the steps finished
        # processing the batches that were sent before the `_BATCH_STOP_FLAG`. Then send
        # the `_BATCH_STOP_FLAG` to the output queue to notify the pipeline to stop.
        while self.output_queue.qsize() != 0:
            pass
        self.shared_info[_STEPS_LOADED_KEY] = _STEPS_LOADED_ERROR_CODE
        self._logger.info("🛑 Stopping pipeline...")
        self.output_queue.put(_BATCH_STOP_FLAG)

    def _handle_keyboard_interrupt(self) -> None:
        """Handles KeyboardInterrupt signal sent during the Pipeline.run method.

        It will try to call self._stop (if the pipeline didn't started yet, it won't
        have any effect), and if the pool is already started, will close it before exiting
        the program.
        """

        def signal_handler(signumber: int, frame: Any) -> None:
            self._logger.info(
                "🚨 CTRL+C signal, waiting steps to finish processing and stopping"
                " pipeline..."
            )
            self._stop()

        signal.signal(signal.SIGINT, signal_handler)


class _ProcessWrapperException(Exception):
    """Exception to be raised when an error occurs in the `Step` process.

    Attributes:
        message: The error message.
        step: The `Step` that raised the error.
        code: The error code.
    """

    def __init__(self, message: str, step: "Step", code: int) -> None:
        self.message = message
        self.step = step
        self.code = code

    @classmethod
    def create_load_error(
        cls, message: str, step: "Step"
    ) -> "_ProcessWrapperException":
        """Creates a `_ProcessWrapperException` for a load error.

        Args:
            message: The error message.
            step: The `Step` that raised the error.

        Returns:
            The `_ProcessWrapperException` instance.
        """
        return cls(message, step, 1)

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
        if isinstance(self.step, _Task) and isinstance(
            self.step.llm, CudaDevicePlacementMixin
        ):
            self.step.llm.set_device_placement_info(
                llm_identifier=self.step.name,
                device_llm_placement_map=self.shared_info[
                    _CUDA_LLM_DEVICE_PLACEMENT_KEY
                ],
                device_llm_placement_lock=self.shared_info[
                    _CUDA_LLM_DEVICE_PLACEMENT_LOCK
                ],
            )

    def run(self) -> None:
        """The target function executed by the process. This function will also handle
        the step lifecycle, executing first the `load` function of the `Step` and then
        waiting to receive a batch from the `input_queue` that will be handled by the
        `process` method of the `Step`.
        """

        # Ignore KeyboardInterrupt signals in the process
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            self.step._logger.debug(f"Loading step '{self.step.name}'...")
            self.step.load()
            self.step._logger.debug(f"Step '{self.step.name}' loaded!")
        except Exception as e:
            raise _ProcessWrapperException.create_load_error(str(e), self.step) from e

        self._notify_load()

        if self.step.is_generator:
            self._generator_step_process_loop()
        else:
            self._non_generator_process_loop()

        self.step._logger.info(f"🏁 Finished running step '{self.step.name}'")

    def _notify_load(self) -> None:
        """Notifies that the step has finished executing its `load` function successfully."""
        with self.shared_info[_STEPS_LOADED_LOCK]:
            self.shared_info[_STEPS_LOADED_KEY] += 1

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
            if (batch := self.input_queue.get()) == _BATCH_STOP_FLAG:
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
                if (batch := self.input_queue.get()) == _BATCH_STOP_FLAG:
                    self.step._logger.info(
                        f"🛑 Stopping yielding batches from step '{self.step.name}'"
                    )
                    return
        except Exception as e:
            raise _ProcessWrapperException(str(e), self.step, 2) from e

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
            if (batch := self.input_queue.get()) == _BATCH_STOP_FLAG:
                self.step._logger.info(
                    f"🛑 Stopping processing batches from step '{self.step.name}'"
                )
                break

            self.step._logger.info(
                f"📦 Processing batch {batch.seq_no} in '{batch.step_name}'"
            )
            # `result` is initally an empty list so f `process` method raises an exception
            # an empty batch will be sent to the `output_queue`
            result = []
            try:
                if self.step.has_multiple_inputs:
                    result = next(self.step.process_applying_mappings(*batch.data))
                else:
                    result = next(self.step.process_applying_mappings(batch.data[0]))
            except Exception as e:
                if self.step.is_global:
                    raise _ProcessWrapperException(str(e), self.step, 2) from e

                # if the step is not global then we can skip the batch which means sending
                # an empty batch to the output queue
                self.step._logger.warning(
                    f"⚠️ Processing batch {batch.seq_no} with step '{self.step.name}' failed:"
                    f" {e}. Sending empty batch..."
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
