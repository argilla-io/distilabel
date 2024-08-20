#  Copyright 2021-present, the Recognai S.L. team.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import dataclasses
import json
import logging
import platform
from typing import TYPE_CHECKING

from huggingface_hub.utils import send_telemetry

from distilabel import __version__
from distilabel.utils.telemetry import (
    is_running_on_docker_container,
)

if TYPE_CHECKING:
    from distilabel.pipeline.base import BasePipeline
    from distilabel.pipeline.batch import _Batch
    from distilabel.steps.base import _Step


_LOGGER: logging.Logger = logging.getLogger(__name__)



@dataclasses.dataclass
class TelemetryClient:
    def __post_init__(self, enable_telemetry: bool):
        self._system_info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "sys_version": platform.version(),
            "docker": is_running_on_docker_container(),
        }

        _LOGGER.info("System Info:")
        _LOGGER.info(f"Context: {json.dumps(self._system_info, indent=2)}")
        self.enable_telemetry = enable_telemetry

    def track_add_step_data(self, pipeline: "BasePipeline", step: "_Step"):
        user_agent = {}

        user_agent["pipeline"] = pipeline.__name__
        user_agent["step"] = step.__name__
        if hasattr(step, "llm"):
            user_agent["llm"] = step.llm.__name__
            topic = "add_task"
        else:
            topic = "add_step"

        self.track_data(topic=topic, user_agent=user_agent)

    def track_add_edge_data(self, pipeline: "BasePipeline", from_step: "_Step", to_step: "_Step"):
        user_agent = {}

        user_agent["pipeline"] = pipeline.__name__
        user_agent["from_step"] = from_step.__name__
        user_agent["to_step"] = to_step.__name__

        self.track_data(topic="add_edge", user_agent=user_agent)

    def track_process_batch_data(self, pipeline: "BasePipeline", step: "_Step", batch: "_Batch"):
        user_agent = {

        }
        user_agent["pipeline"] = pipeline.__name__
        user_agent["step"] = step.__name__
        user_agent["batch_size"] = batch.size

        self.track_data(topic="process_batch", user_agent=user_agent)

    def track_data(self, topic: str, user_agent: dict, include_system_info: bool = True):
        library_name = "distilabel"
        topic = f"{library_name}/{topic}"

        if include_system_info:
            user_agent.update(self._system_info)

        send_telemetry(topic=topic, library_name=library_name, library_version=__version__, user_agent=user_agent)


_TELEMETRY_CLIENT = TelemetryClient()


def get_telemetry_client() -> TelemetryClient:
    return _TELEMETRY_CLIENT
