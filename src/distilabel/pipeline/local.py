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

import json
import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Set, cast

from distilabel.pipeline.base import BasePipeline, _Batch, _BatchManager
from distilabel.steps.base import Step

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager
    from multiprocessing.pool import Pool
    from os import PathLike
    from queue import Queue

    from distilabel.steps.base import GeneratorStep


class Pipeline(BasePipeline):
    """Local pipeline implementation using `multiprocessing`."""

    def run(self, parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Runs the pipeline.

        Args:
            parameters: a dictionary containing the runtime parameters for each step.
                The keys are the step names and the values are dictionaries in which the
                keys are the parameter names (defined in the `process` method of the step)
                and the values are the parameter values.
        """
        super().run(parameters)

        leaf_steps_received_last_batch = {
            step_name: False for step_name in self.dag.leaf_steps
        }

        buffer_data_path = self._cache_filenames["data"]
        self._logger.info(f"📝 Writing buffer to {buffer_data_path}")
        write_buffer = _WriteBuffer(
            path=buffer_data_path, leaf_steps=self.dag.leaf_steps
        )
        if self._batch_manager is None:
            self._batch_manager = _BatchManager.from_dag(self.dag)

        ctx = mp.get_context("forkserver")
        with ctx.Manager() as manager, ctx.Pool(mp.cpu_count()) as pool:
            output_queue: "Queue[_Batch]" = manager.Queue()
            self._run_steps_in_loop(pool, manager, output_queue)

            # TODO: Update this to run only on fresh _BAtchManager
            self._request_initial_batches()

            # TODO: write code for handling output batch to new method and write unit test
            while True:
                batch = output_queue.get()

                for step_name in self.dag.get_step_successors(batch.step_name):
                    for new_batch in self._batch_manager.add_batch(
                        to_step=step_name, batch=batch
                    ):
                        self._send_batch_to_step(new_batch)

                # If step is generator and previous batch was not the last one, then request
                # next batch to the generator step
                if not batch.last_batch:
                    step = self.dag.get_step(batch.step_name)["step"]
                    if step.is_generator:
                        self._send_batch_to_step(batch.next_batch())

                if batch.step_name in self.dag.leaf_steps:
                    write_buffer.add_batch(batch.step_name, batch)

                    if batch.last_batch:
                        leaf_steps_received_last_batch[batch.step_name] = True

                    # All the leaf steps have processed the last batch, stop the generation
                    if all(leaf_steps_received_last_batch.values()):
                        break

    def _request_initial_batches(self) -> None:
        """Requests the initial batches to the generator steps."""
        for step_name in self.dag.root_steps:
            batch = _Batch(seq_no=0, step_name=step_name, last_batch=False)
            self._send_batch_to_step(batch)

    def _send_batch_to_step(self, batch: "_Batch") -> None:
        """Sends a batch to the input queue of a step.

        Args:
            batch: The batch to send.
        """
        input_queue = self.dag.get_step(batch.step_name)["input_queue"]
        input_queue.put(batch)

    def _run_steps_in_loop(
        self, pool: "Pool", manager: "SyncManager", output_queue: "Queue[_Batch]"
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
        """
        for step_name in self.dag:
            step = self.dag.get_step(step_name)["step"]
            input_queue = manager.Queue()
            self.dag.set_step_attr(step.name, "input_queue", input_queue)

            process_wrapper = _ProcessWrapper(
                step=step, input_queue=input_queue, output_queue=output_queue
            )

            pool.apply_async(process_wrapper.run, error_callback=self._error_callback)  # type: ignore

    def _error_callback(self, e: "_ProcessWrapperException") -> None:
        """Error callback that will be called when an error occurs in a `Step` process.

        Args:
            e: The `_ProcessWrapperException` containing the error message and the `Step`
                that raised the error.
        """
        # TODO: handle the errors in a better way
        # TODO: Save state of the pipeline at this stage
        self._logger.error(f"ERROR: {e}")


class _WriteBuffer:
    def __init__(self, path: "PathLike", leaf_steps: Set[str]) -> None:
        self._path = Path(path)
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffers: Dict[str, Any] = {step: None for step in leaf_steps}

    @property
    def is_full(self) -> bool:
        return all(self._buffers.values())

    def add_batch(self, step_name: str, batch: "_Batch") -> None:
        self._buffers[step_name] = batch.data
        if self.is_full:
            self._write()

    def _write(self) -> None:
        data = list(self._combine_batches())

        with open(self._path, "a") as f:
            for rows in data:
                for row in rows:
                    json.dump(row, f)
                    f.write("\n")

        self._clean_buffers()

    def _combine_batches(self) -> Iterator[Dict[str, Any]]:
        for _, data in self._buffers.items():
            yield data[-1]

    def _clean_buffers(self) -> None:
        self._buffers = {step: None for step in self._buffers.keys()}


class _ProcessWrapperException(Exception):
    """Exception to be raised when an error occurs in the `Step` process.

    Attributes:
        message: The error message.
        step: The `Step` that raised the error.
    """

    def __init__(self, message: str, step: "Step") -> None:
        self.message = message
        self.step = step


class _ProcessWrapper:
    """Wrapper to run the `Step` in a separate process.

    Attributes:
        step: The step to run.
        input_queue: The queue to receive the input data.
        output_queue: The queue to send the output data.
    """

    def __init__(
        self, step: "Step", input_queue: "Queue[_Batch]", output_queue: "Queue[_Batch]"
    ) -> None:
        """Initializes the `_ProcessWrapper`.

        Args:
            step: The step to run.
            input_queue: The queue to receive the input data.
            output_queue: The queue to send the output data.
        """
        self.step = step
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self) -> None:
        """The target function executed by the process. This function will also handle
        the step lifecycle, executing first the `load` function of the `Step` and then
        waiting to receive a batch from the `input_queue` that will be handled by the
        `process` method of the `Step`.
        """

        def _run() -> None:
            self.step.load()

            batch = self.input_queue.get()
            if self.step.is_generator:
                self._process_generator_step(batch)
            else:
                while True:
                    self._process_non_generator_step(batch)
                    if batch.last_batch:
                        break
                    batch = self.input_queue.get()
            self.step._logger.info(f"🏁 Finished running step '{self.step.name}'")

        try:
            _run()
        except Exception as e:
            raise _ProcessWrapperException(str(e), self.step) from e

    def _process_generator_step(self, batch: _Batch) -> None:
        """Processes a batch in a generator step. It will call the `process` method of the
        step and send the output data to the `output_queue` and block until the next batch
        request is received (i.e. receiving an empty batch from the `input_queue`).

        If the `last_batch` attribute of the batch is `True`, the loop will stop and the
        process will finish.

        Args:
            batch: The batch to process.
        """
        step = cast("GeneratorStep", self.step)
        self.step._logger.info(
            f"🧬 Starting yielding batches from generator step '{batch.step_name}'"
        )
        for data, last_batch in step.process_applying_mappings():
            batch.data = [data]
            batch.last_batch = last_batch
            self.output_queue.put(batch)
            if batch.last_batch:
                self.step._logger.info(
                    f"🏁 Finished yielding batches from generator step '{batch.step_name}'"
                )
                return
            batch = self.input_queue.get()

    def _process_non_generator_step(self, batch: _Batch) -> None:
        """Processes a batch in a non-generator step. It will call the `process` method
        of the step and send the output data to the `output_queue`. It will take care of
        the case when the step has multiple inputs.

        Args:
            batch: The batch to process.
        """
        if self.step.is_global:
            self.step._logger.info(f"🌍 Running global step '{batch.step_name}'")
        else:
            self.step._logger.info(
                f"📦 Processing batch {batch.seq_no} in '{batch.step_name}'"
            )

        if self.step.has_multiple_inputs:
            result = next(self.step.process_applying_mappings(*batch.data))
        else:
            result = next(self.step.process_applying_mappings(batch.data[0]))
        self.step._logger.info(
            f"📨 Step '{batch.step_name}' sending batch {batch.seq_no} to output queue"
        )
        batch.data = [result]
        self.output_queue.put(batch)
