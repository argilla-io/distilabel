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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

from distilabel.pipeline.base import BasePipeline
from distilabel.pipeline.step.base import Step

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager
    from multiprocessing.pool import Pool
    from queue import Queue

    from distilabel.pipeline._dag import DAG
    from distilabel.pipeline.step.base import GeneratorStep


class Pipeline(BasePipeline):
    """Local pipeline implementation using `multiprocessing`."""

    def run(self, parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        super().run(parameters)

        leaf_steps_received_last_batch = {
            step_name: False for step_name in self.dag.leaf_steps
        }

        write_buffer = _WriteBuffer(path="./data.jsonl", leaf_steps=self.dag.leaf_steps)
        batch_manager = _BatchManager.from_dag(self.dag)

        ctx = mp.get_context("forkserver")
        with ctx.Manager() as manager, ctx.Pool(mp.cpu_count()) as pool:
            output_queue: "Queue[_Batch]" = manager.Queue()
            self._create_processes(pool, manager, output_queue)

            # Send initial empty batch to trigger the batch flow between the steps
            for step_name in self.dag.root_steps:
                self._request_batch_to_generator(step_name)

            while True:
                batch = output_queue.get()

                for step_name in self.dag.get_step_successors(batch.step_name):
                    if batches := batch_manager.add_batch(
                        to_step=step_name, from_step=batch.step_name, batch=batch
                    ):
                        data = [batch.data[0] for batch in batches]
                        self._send_batch_to_step(
                            step_name=step_name,
                            batch=_Batch(
                                step_name=step_name,
                                last_batch=batch.last_batch,
                                data=data,
                            ),
                        )

                # If step is generator and previous batch was not the last one, then request
                # next batch to the generator step
                if not batch.last_batch:
                    step = self.dag.get_step(batch.step_name)["step"]
                    if step.is_generator:
                        self._request_batch_to_generator(batch.step_name)

                if batch.step_name in self.dag.leaf_steps:
                    write_buffer.add_batch(batch.step_name, batch)

                    if batch.last_batch:
                        leaf_steps_received_last_batch[batch.step_name] = True

                    # All the leaf steps have processed the last batch, stop the generation
                    if all(leaf_steps_received_last_batch.values()):
                        break

    def _send_batch_to_step(self, step_name: str, batch: "_Batch") -> None:
        input_queue = self.dag.get_step(step_name)["input_queue"]
        input_queue.put(
            _Batch(step_name=step_name, last_batch=batch.last_batch, data=batch.data)
        )

    def _request_batch_to_generator(self, step_name: str) -> None:
        self._send_batch_to_step(
            step_name, _Batch(step_name=step_name, last_batch=False)
        )

    def _create_processes(
        self, pool: "Pool", manager: "SyncManager", output_queue: "Queue[_Batch]"
    ) -> None:
        for step_name in self.dag:
            step = self.dag.get_step(step_name)["step"]
            input_queue = manager.Queue()
            self.dag.set_step_attr(step.name, "input_queue", input_queue)

            process_wrapper = _ProcessWrapper(
                step=step, input_queue=input_queue, output_queue=output_queue
            )

            pool.apply_async(process_wrapper.run, error_callback=self._error_callback)

    def _error_callback(self, e: Exception) -> None:
        print("ERROR", e)


@dataclass
class _Batch:
    """Dataclass to represent a batch of data to be processed by a `Step`.

    Attributes:
        step_name: The name of the step that will process the batch.
        last_batch: A flag to indicate if the batch is the last one.
        data: The data to be processed.
    """

    step_name: str
    last_batch: bool
    data: List[List[Dict[str, Any]]] = field(default_factory=list, repr=False)


class _BatchManager:
    """Class to manage the batches received from the steps. It keeps track of the
    received batches and returns the list of batches to be processed when all the inputs
    for a step are received.

    Attributes:
        _batches: A dictionary with the step name as the key and a dictionary with the
        predecessor step name as the key and the batch as the value.
    """

    def __init__(self, batches: Dict[str, Dict[str, Union["_Batch", None]]]) -> None:
        self._batches = batches

    def add_batch(
        self, to_step: str, from_step: str, batch: _Batch
    ) -> Union[List[_Batch], None]:
        """Add an output batch from `from_step` to `to_step`. If all the inputs for
        `to_step` are received, then return the list of batches to be processed.

        Args:
            to_step: The name of the step that will process the batch.
            from_step: The name of the step that generated the batch.
            batch: The output batch to be added to `to_step` from `from_step`.

        Returns:
            If all the inputs for `to_step` are received, then return the list of batches
            to be processed. Otherwise, return `None`.
        """

        if self._batches[to_step][from_step] is not None:
            raise ValueError(
                f"Step '{to_step}' already had a batch waiting from '{from_step}'"
            )
        self._batches[to_step][from_step] = batch

        if self._step_input_batches_received(to_step):
            batches = []
            for batch in self._batches[to_step].values():  # type: ignore
                batches.append(batch)
            self._clean_step_input_batches(to_step)
            return batches

        return None

    @classmethod
    def from_dag(cls, dag: "DAG") -> "_BatchManager":
        """Create a `_BatchManager` instance from a `DAG` instance.

        Args:
            dag: The `DAG` instance.

        Returns:
            A `_BatchManager` instance.
        """
        batches = {}
        for step_name in dag:
            # Skip generator steps as they don't have predecessors
            if dag.get_step(step_name)["step"].is_generator:
                continue
            batches[step_name] = {}
            for predecessor in dag.get_step_predecessors(step_name):
                batches[step_name][predecessor] = None
        return cls(batches)

    def _step_input_batches_received(self, step_name: str) -> bool:
        """Check if all the input batches for a step have been received.

        Args:
            step_name: The name of the step.

        Returns:
            A boolean indicating if all the inputs for the step have been received.
        """

        return all(self._batches[step_name].values())

    def _clean_step_input_batches(self, step_name: str) -> None:
        """Clean the input batches for a step.

        Args:
            step_name: The name of the step.
        """
        self._batches[step_name] = {step: None for step in self._batches[step_name]}


class _WriteBuffer:
    """Buffer to store a batch from each leaf step until the buffer. When the buffer is
    full, the batches will be combined and appended to a JSONL file, and the buffer will
    be cleaned.

    Attributes
        _buffers: A dictionary with the step name as the key and the batch as the value.
    """

    def __init__(self, path, leaf_steps: List[str]) -> None:
        self._path = path
        self._buffers: Dict[str, Any] = {step: None for step in leaf_steps}

    @property
    def is_full(self) -> bool:
        return all(self._buffers.values())

    def add_batch(self, step_name: str, batch: "_Batch") -> None:
        self._buffers[step_name] = batch.data
        if self.is_full:
            self._write()

    def _write(self) -> None:
        data = self._combine_batches()

        with open(self._path, "a") as f:
            for row in data:
                json.dump(row, f)
                f.write("\n")

        self._clean_buffers()

    def _combine_batches(self) -> List[Dict[str, Any]]:
        for _, data in self._buffers.items():
            return data

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

            while True:
                batch = self.input_queue.get()
                if self.step.is_generator:
                    self._process_generator_step(batch)
                    break

                self._process_non_generator_step(batch)
                if batch.last_batch:
                    break

        try:
            _run()
        except Exception as e:
            raise _ProcessWrapperException(str(e), self.step) from e

    def _process_generator_step(self, batch: _Batch) -> None:
        step = cast("GeneratorStep", self.step)
        for data, last_batch in step.process(**self.step._runtime_parameters):
            batch.data = [data]
            batch.last_batch = last_batch
            self.output_queue.put(batch)
            if batch.last_batch:
                return
            batch = self.input_queue.get()

    def _process_non_generator_step(self, batch: _Batch) -> None:
        if self.step.has_multiple_inputs:
            result = next(
                self.step.process(*batch.data, **self.step._runtime_parameters)
            )
        else:
            result = next(
                self.step.process(batch.data[0], **self.step._runtime_parameters)
            )
        batch.data = [result]
        self.output_queue.put(batch)
