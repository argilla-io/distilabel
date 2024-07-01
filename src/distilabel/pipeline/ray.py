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

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from distilabel.distiset import create_distiset
from distilabel.pipeline.base import BasePipeline
from distilabel.utils.logging import stop_logging

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

        # TODO: check if ray queue can be used
        self._set_logging_parameters(
            {
                "log_queue": self.QueueClass(),
                "filename": self._cache_location["log_file"],
            }
        )

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
        try:
            import ray
        except ImportError as ie:
            raise ImportError(
                "ray is not installed. Please install it using `pip install ray`."
            ) from ie

        if self._ray_head_node_host:
            ray.init(f"ray://{self._ray_head_node_host}:{self._ray_head_node_port}")
        else:
            ray.init()

    @property
    def QueueClass(self) -> Callable:
        from ray.util.queue import Queue

        return Queue

    def _run_step(self, step: "_Step", input_queue: "Queue[Any]", replica: int) -> None:
        pass

    def _teardown(self) -> None:
        if self._output_queue:
            self._output_queue.shutdown()

        if self._load_queue:
            self._load_queue.shutdown()

    def _set_steps_not_loaded_exception(self) -> None:
        pass

    def _stop(self) -> None:
        pass
