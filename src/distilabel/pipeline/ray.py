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

from distilabel.distiset import create_distiset
from distilabel.pipeline.base import BasePipeline
from distilabel.pipeline.step_wrapper import _StepWrapper
from distilabel.utils.logging import setup_logging, stop_logging

if TYPE_CHECKING:
    from os import PathLike
    from queue import Queue

    from distilabel.distiset import Distiset
    from distilabel.steps.base import _Step


class RayPipeline(BasePipeline):
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        cache_dir: Optional[Union[str, "PathLike"]] = None,
        enable_metadata: bool = False,
        requirements: Optional[List[str]] = None,
        ray_head_node_host: Optional[str] = None,
        ray_head_node_port: int = 10001,
    ) -> None:
        super().__init__(name, description, cache_dir, enable_metadata, requirements)

        self._ray_head_node_host = ray_head_node_host
        self._ray_head_node_port = ray_head_node_port

    def run(
        self,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache: bool = True,
        storage_parameters: Optional[Dict[str, Any]] = None,
        use_fs_to_pass_data: bool = False,
    ) -> "Distiset":
        self._init_ray()

        self._log_queue = self.QueueClass()

        if distiset := super().run(
            parameters, use_cache, storage_parameters, use_fs_to_pass_data
        ):
            return distiset

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
            stop_logging()
            raise self._exception

        distiset = create_distiset(
            self._cache_location["data"],
            pipeline_path=self._cache_location["pipeline"],
            log_filename_path=self._cache_location["log_file"],
            enable_metadata=self._enable_metadata,
        )

        stop_logging()

        return distiset

    def _init_ray(self) -> None:
        """Init or connects to a Ray cluster."""
        try:
            import ray
        except ImportError as ie:
            raise ImportError(
                "ray is not installed. Please install it using `pip install ray`."
            ) from ie

        if self._ray_head_node_host:
            ray.init(
                f"ray://{self._ray_head_node_host}:{self._ray_head_node_port}",
                runtime_env={"pip": self.requirements},
            )
        else:
            ray.init()

    @property
    def QueueClass(self) -> Callable:
        from ray.util.queue import Queue

        return Queue

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
                self, step_wrapper: _StepWrapper, log_queue: "Queue[Any]"
            ) -> None:
                self._step_wrapper = step_wrapper
                self._log_queue = log_queue

            def run(self) -> str:
                setup_logging(log_queue=self._log_queue)
                return self._step_wrapper.run()

        step_wrapper = _StepWrapperRay.remote(
            step_wrapper=_StepWrapper(
                step=step,  # type: ignore
                replica=replica,
                input_queue=input_queue,
                output_queue=self._output_queue,
                load_queue=self._load_queue,
                dry_run=self._dry_run,
            ),
            log_queue=self._log_queue,
        )

        resources: Dict[str, Any] = {
            "name": f"distilabel-{self.name}-{step.name}-{replica}"
        }

        if step.resources.cpus is not None:
            resources["num_cpus"] = step.resources.cpus

        if step.resources.gpus is not None:
            resources["num_gpus"] = step.resources.gpus

        step_wrapper.options(**resources).remote()

        step_wrapper.run.remote()

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

        self._stop_load_queue_loop()
        self._stop_output_queue_loop()
