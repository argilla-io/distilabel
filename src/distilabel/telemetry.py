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

import dataclasses
import platform
from typing import TYPE_CHECKING, Optional, Union

from huggingface_hub.utils import send_telemetry

from distilabel import __version__
from distilabel.utils.telemetry import (
    is_running_on_docker_container,
)

if TYPE_CHECKING:
    from distilabel.pipeline.base import BasePipeline
    from distilabel.pipeline.batch import _Batch
    from distilabel.steps.base import _Step


@dataclasses.dataclass
class TelemetryClient:
    def __post_init__(self):
        self._system_info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "sys_version": platform.version(),
            "docker": is_running_on_docker_container(),
        }

    def track_add_step_data(self, pipeline: "BasePipeline", step: "_Step"):
        user_agent = {}

        user_agent["pipeline"] = pipeline.__class__.__name__
        user_agent["step"] = step.__class__.__name__
        if hasattr(step, "llm"):
            user_agent["llm"] = step.llm.__class__.__name__
            user_agent["task"] = step.__class__.__name__
        else:
            user_agent["step"] = step.__class__.__name__

        if step.is_generator:
            user_agent["type"] = "generator"
        elif step.is_global:
            user_agent["type"] = "global"
        else:
            user_agent["type"] = "normal"

        self._track_data(topic="add_step", user_agent=user_agent)

    def track_add_edge_data(
        self, pipeline: "BasePipeline", from_step: "_Step", to_step: "_Step"
    ):
        user_agent = {}

        user_agent["pipeline"] = pipeline.__class__.__name__
        user_agent["from_step"] = from_step.__class__.__name__
        user_agent["to_step"] = to_step.__class__.__name__

        self._track_data(topic="add_edge", user_agent=user_agent)

    def track_process_batch_data(
        self, pipeline: "BasePipeline", step: "_Step", batch: "_Batch"
    ):
        user_agent = {}

        user_agent["pipeline"] = pipeline.__class__.__name__
        user_agent["step"] = step.__class__.__name__
        user_agent["batch_size"] = batch.size

        self._track_data(topic="process_batch", user_agent=user_agent)

    def track_run_data(self, pipeline: "BasePipeline", user_agent: dict):
        user_agent["pipeline"] = pipeline.__class__.__name__

        self._track_data(topic="run", user_agent=user_agent)

    def track_exception(
        self, pipeline: "BasePipeline", exception: Union[Exception, str]
    ):
        user_agent = {}

        user_agent["pipeline"] = pipeline.__class__.__name__
        if isinstance(exception, Exception):
            user_agent["exception"] = exception.__class__.__name__
        else:
            user_agent["exception"] = exception

        self._track_data(topic="exception", user_agent=user_agent)

    def _track_data(self, topic: str, user_agent: Optional[dict] = None):
        library_name = "distilabel"
        topic = f"{library_name}/{topic}"

        user_agent = user_agent or {}
        user_agent.update(self._system_info)

        send_telemetry(
            topic=topic,
            library_name=library_name,
            library_version=__version__,
            user_agent=user_agent,
        )


_TELEMETRY_CLIENT = TelemetryClient()


def get_telemetry_client() -> TelemetryClient:
    return _TELEMETRY_CLIENT
