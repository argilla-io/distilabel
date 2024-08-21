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

import importlib.resources as importlib_resources
from typing import TYPE_CHECKING, Any, Dict, Union

from jinja2 import Template

from distilabel.steps.tasks import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


class URIAL(Task):
    def load(self) -> None:
        """Loads the Jinja2 template for the given `aspect`."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "ultrafeedback"
            / "urial.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> "StepColumns":
        return {"instruction": False, "conversation": False}

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        messages = (
            [{"role": "user", "content": input["instruction"]}]
            if "instruction" in input
            else input["conversation"]
        )

        if messages[-1]["role"] != "user":
            raise ValueError("The last message must be from the user.")

        return [{"role": "user", "content": self._template.render(messages=messages)}]

    @property
    def outputs(self) -> "StepColumns":
        return ["generation", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        if output is None:
            return {}
        pass
