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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from distilabel.pipeline.base import BasePipeline

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager
    from multiprocessing.pool import Pool
    from queue import Queue

    from distilabel.step.base import GeneratorStep, Step


def error_callback(e: Exception) -> None:
    print("ERROR", e)


class Pipeline(BasePipeline):
    """Local pipeline implementation using `multiprocessing`."""

    def run(self, parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        super().run(parameters)

        leaf_steps_received_last_batch = {
            step_name: False for step_name in self.dag.leaf_steps
        }

        write_buffer = _WriteBuffer(path="./data.jsonl", leaf_steps=self.dag.leaf_steps)

        ctx = mp.get_context("forkserver")
        with ctx.Manager() as manager, ctx.Pool(mp.cpu_count()) as pool:
            output_queue: "Queue[_Batch]" = manager.Queue()
            self._create_processes(pool, manager, output_queue)

            # Send initial empty batch to trigger the batch flow between the steps
            for step_name in self.dag.root_steps:
                self._request_batch_to_generator(step_name)

            while True:
                batch = output_queue.get()

                # TODO: oversimplified for now, it works for fully sequential pipelines
                # Things to consider:
                #   - Step receiving output from more than one step
                #   - Global steps that needs to received all the data at once
                for step_name in self.dag.get_step_successors(batch.step_name):
                    self._send_batch_to_step(
                        step_name=step_name,
                        batch=_Batch(
                            step_name=step_name,
                            last_batch=batch.last_batch,
                            data=batch.data,
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
            _Batch(
                step_name=step_name,
                last_batch=batch.last_batch,
                data=batch.data,
            )
        )

    def _request_batch_to_generator(self, step_name: str) -> None:
        self._send_batch_to_step(
            step_name, _Batch(step_name=step_name, last_batch=False)
        )

    def _create_processes(
        self, pool: "Pool", manager: "SyncManager", output_queue: "Queue[_Batch]"
    ) -> None:
        for step_name in self.dag:  # type: ignore
            step = self.dag.get_step(step_name)["step"]
            input_queue = manager.Queue()
            self.dag.set_step_attr(step.name, "input_queue", input_queue)

            process_wrapper = _ProcessWrapper(
                step=step, input_queue=input_queue, output_queue=output_queue
            )

            pool.apply_async(process_wrapper.run, error_callback=error_callback)


@dataclass
class _Batch:
    """Class containing"""

    step_name: str
    last_batch: bool
    data: List[Dict[str, Any]] = field(default_factory=list, repr=False)


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

        self.step.load()

        while True:
            batch = self.input_queue.get()
            if self.step.is_generator:
                self._process_generator_step(batch)
                break

            self._process_non_generator_step(batch)
            if batch.last_batch:
                break

    def _process_generator_step(self, batch: _Batch) -> None:
        step = cast("GeneratorStep", self.step)
        for data, last_batch in step.process(**self.step._runtime_parameters):
            batch.data = data
            batch.last_batch = last_batch
            self.output_queue.put(batch)
            if batch.last_batch:
                return
            batch = self.input_queue.get()

    def _process_non_generator_step(self, batch: _Batch) -> None:
        result = next(self.step.process(batch.data, **self.step._runtime_parameters))
        batch.data = result
        self.output_queue.put(batch)
