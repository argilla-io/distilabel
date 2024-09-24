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
from distilabel.telemetry._helpers import get_server_id
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
            "server_id": get_server_id(),
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "sys_version": platform.version(),
            "docker": is_running_on_docker_container(),
        }

    def track_add_step_data(
        self, pipeline: "BasePipeline", step: str, step_type: str, llm: str = None
    ):
        user_agent = {}

        user_agent["pipeline"] = pipeline.__class__.__name__
        user_agent["step"] = step
        user_agent["llm"] = llm
        user_agent["type"] = step_type

        self._track_data(topic="add_step", user_agent=user_agent)

    def track_process_batch_data(
        self, pipeline: "BasePipeline", step: "_Step", batch: "_Batch", is_leaf: bool
    ):
        user_agent = {}
        user_agent["is_leaf"] = is_leaf
        user_agent["pipeline_id"] = pipeline._create_signature()
        user_agent["pipeline"] = pipeline.__class__.__name__
        user_agent["step"] = step.__class__.__name__
        user_agent["batch_size"] = batch.size

        self._track_data(topic="process_batch", user_agent=user_agent)

    def track_run_data(self, pipeline: "BasePipeline", user_agent: dict):
        # Get the steps and connections from the pipeline dump
        dump = pipeline.dump()
        steps = dump["pipeline"]["steps"]
        step_name_to_type = {
            step["step"].get("name"): step["step"].get("type_info", {}).get("name")
            for step in steps
        }
        step_name_to_class = {
            step["step"].get("name"): step["step"].get("type_info", {}).get("name")
            for step in steps
        }
        step_name_to_llm = {
            step["step"].get("name"): step["step"].get("llm") for step in steps
        }
        step_name_to_llm = {k: v for k, v in step_name_to_llm.items() if v is not None}

        # Track the steps
        for step in step_name_to_type:
            llm = step_name_to_llm.get(step)
            step_type = "task" if llm is not None else "step"
            self.track_add_step_data(pipeline, step_name_to_class[step], step_type, llm)

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
