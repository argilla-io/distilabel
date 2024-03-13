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
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    cast,
)

import pyarrow as pa
import pyarrow.parquet as pq

from distilabel.pipeline.base import BasePipeline, _Batch, _BatchManager
from distilabel.steps.base import Step

if TYPE_CHECKING:
    from multiprocessing.managers import DictProxy, SyncManager
    from multiprocessing.pool import Pool
    from os import PathLike
    from queue import Queue

    from distilabel.steps.base import GeneratorStep

_STEPS_LOADED_LOCK_KEY = "lock"
_STEPS_LOADED_KEY = "steps_loaded"
_STEPS_LOADED_ERROR_CODE = -1


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
        self._logger.info("📝 Writing buffer to cache folder")
        write_buffer = _WriteBuffer(
            path=buffer_data_path, leaf_steps=self.dag.leaf_steps
        )
        if self._batch_manager is None:
            self._batch_manager = _BatchManager.from_dag(self.dag)

        ctx = mp.get_context("forkserver")
        with ctx.Manager() as manager, ctx.Pool(mp.cpu_count()) as pool:
            self.output_queue: "Queue[Any]" = manager.Queue()
            self.shared_info = manager.dict(
                **{_STEPS_LOADED_KEY: 0, _STEPS_LOADED_LOCK_KEY: manager.Lock()}
            )

            # Run the steps using the pool of processes
            self._run_steps_in_loop(pool, manager, self.output_queue, self.shared_info)

            # Wait for all the steps to be loaded correctly
            if not self._all_steps_loaded():
                return

            self._request_initial_batches()

            # TODO: write code for handling output batch to new method and write unit test
            while True:
                batch = self.output_queue.get()

                # If `None` is received, then stop the pipeline
                if batch is None:
                    break

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

    def _all_steps_loaded(self) -> bool:
        """Waits for all the steps to load.

        Returns:
            `True` if all the steps have been loaded correctly, `False` otherwise.
        """
        self._logger.info("⏳ Waiting for all the steps to load...")
        while True:
            with self.shared_info[_STEPS_LOADED_LOCK_KEY]:
                steps_loaded = self.shared_info[_STEPS_LOADED_KEY]

                if steps_loaded == len(self.dag):
                    self._logger.info("✅ All the steps have been loaded!")
                    return True

                if steps_loaded == _STEPS_LOADED_ERROR_CODE:
                    self._logger.error("❌ Failed to load all the steps")
                    return False

                self._logger.info(f"⏳ Steps loaded: {steps_loaded}/{len(self.dag)}")

            time.sleep(5)

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

            pool.apply_async(
                process_wrapper.run,
                error_callback=self._error_callback,
                callback=lambda _: self._cache(),
            )  # type: ignore

    def _error_callback(self, e: "_ProcessWrapperException") -> None:
        """Error callback that will be called when an error occurs in a `Step` process.

        Args:
            e: The `_ProcessWrapperException` containing the error message and the `Step`
                that raised the error.
        """
        if e.is_load_error:
            self._logger.error(f"Failed to load step '{e.step.name}': {e.message}")
        else:
            self._logger.error(
                f"An error occurred in step '{e.step.name}': {e.message}"
            )
        self._cache()
        self._stop()

    def _stop(self) -> None:
        """Stops the pipeline execution. It will send `None` to the `output_queue` to
        notify the pipeline to stop, and set the `_STEPS_LOADED_KEY` to `_STEPS_LOADED_ERROR_CODE`
        for the pipeline to stop waiting for the steps to load.
        """
        self._logger.info("Stopping pipeline...")
        self.output_queue.put(None)
        with self.shared_info[_STEPS_LOADED_LOCK_KEY]:
            self.shared_info[_STEPS_LOADED_KEY] = _STEPS_LOADED_ERROR_CODE


class _WriteBuffer:
    """Class in charge of sending the batched contents to a buffer and writing
    those to files under a given folder.

    As batches are received, they are added to the buffer and once each buffer
    is full, the content is written to a parquet file.
    """

    _type_map: Dict[type, pa.DataType] = {
        int: pa.int64(),
        float: pa.float64(),
        str: pa.string(),
    }

    def __init__(self, path: "PathLike", leaf_steps: Set[str]) -> None:
        """
        Args:
            path (PathLike): Folder where the files will be written, the idea
                is for this path to be in the cache folder under /data.
            leaf_steps (Set[str]): Leaf steps from the dag of the pipeline.

        Raises:
            ValueError: If the path is not a directory.
        """
        self._path = Path(path)
        if not self._path.is_dir():
            raise ValueError(f"The path should be a directory, not a file: {path}")
        if not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)
        self._buffers: Dict[str, Any] = {step: None for step in leaf_steps}
        self._writers: Dict[str, pq.ParquetWriter] = {}

    def _get_filename(self, step_name: str) -> str:
        """Creates the filename for the step.

        Args:
            step_name (str): Name of the step to which the data pertains.

        Returns:
            Filename for the step.
        """
        return self._path / f"{step_name}.parquet"

    def is_full(self, step_name: str) -> bool:
        """Checks the buffers that are full so that those can be written to the file.

        Returns:
            Whether the buffer is full.
        """
        return bool(self._buffers[step_name])

    def add_batch(self, step_name: str, batch: "_Batch") -> None:
        """Adds a batch to the buffer and writes the buffer to the file if it's full.

        Args:
            step_name (str): Name of the step to which the data pertains.
            batch (_Batch): Batch to add to the buffer.
        """
        self._buffers[step_name] = batch.data
        if self.is_full(step_name):
            self._write(step_name)

    def _write(self, step_name: str) -> None:
        """Writes the content to the file and cleans the buffer.

        Args:
            step_name (str): Name of the step to which the data pertains.
        """
        # TODO: The parquet files should be rotated to different files up to a given size.
        data = self._buffers[step_name]
        writer = self._get_writer(step_name, data)
        for batch in data:
            arrow_batch = pa.RecordBatch.from_pylist(batch)
            writer.write_batch(arrow_batch)

        self._clean_buffer(step_name)

    def _get_writer(
        self, step_name: str, batch_data: List[List[Dict[str, Any]]]
    ) -> pq.ParquetWriter:
        if writer := self._writers.get(step_name):
            return writer
        else:
            filename = self._get_filename(step_name)
            # Get the table schema from the first record in the batch's data.
            schema = pa.schema(
                [
                    pa.field(key, self._type_map[type(value)])
                    for key, value in batch_data[0][0].items()
                ]
            )
            writer = pq.ParquetWriter(filename, schema)
            self._writers[step_name] = writer
            return writer

    def _clean_buffer(self, step_name: str) -> None:
        """Cleans the buffer by setting it's content to None.

        Args:
            step_name (str): The name of the buffer to clean.
        """
        buffs = {}
        for step, data in self._buffers.items():
            if step_name == step:
                buffs[step] = None
            else:
                buffs[step] = data
        self._buffers = buffs

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

    # def _combine_batches(self) -> Iterator[Dict[str, Any]]:
    #     for _, data in self._buffers.items():
    #         print("DATA", data)
    #         yield data[-1]


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

    def run(self) -> None:
        """The target function executed by the process. This function will also handle
        the step lifecycle, executing first the `load` function of the `Step` and then
        waiting to receive a batch from the `input_queue` that will be handled by the
        `process` method of the `Step`.
        """

        try:
            self.step.load()
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
        with self.shared_info["lock"]:
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

        batch = self.input_queue.get()

        self.step._logger.info(
            f"🧬 Starting yielding batches from generator step '{self.step.name}'"
        )

        try:
            for data, last_batch in step.process_applying_mappings():
                batch.data = [data]
                batch.last_batch = last_batch
                self._send_batch(batch)
                if batch.last_batch:
                    return
                batch = self.input_queue.get()
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
            batch = self.input_queue.get()
            self.step._logger.info(
                f"📦 Processing batch {batch.seq_no} in '{batch.step_name}'"
            )
            try:
                if self.step.has_multiple_inputs:
                    result = next(self.step.process_applying_mappings(*batch.data))
                else:
                    result = next(self.step.process_applying_mappings(batch.data[0]))
            except Exception as e:
                if self.step.is_global:
                    raise _ProcessWrapperException(
                        message=str(e), step=self.step, code=2
                    ) from e

                # if the step is not global then we can skip the batch which means sending
                # an empty batch to the output queue
                self.step._logger.warning(
                    f"⚠️ Processing batch {batch.seq_no} with step '{self.step.name}' failed:"
                    f" {e}. Sending empty batch..."
                )
                result = []

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
