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

import re
import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from typing import TYPE_CHECKING, Any, Dict, List, Union

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.typing import ChatType


_PARSE_GENSTRUCT_OUTPUT_REGEX = r"(.+?)\[\[\[Assistant\]\]\](.+)$"


class Genstruct(Task):
    """Generate a pair of instruction-response from a document using an `LLM`.

    `Genstruct` is a pre-defined task designed to generate valid instructions from a given raw document,
    with the title and the content, enabling the creation of new, partially synthetic instruction finetuning
    datasets from any raw-text corpus. The task is based on the Genstruct 7B model by Nous Research, which is
    inspired in the Ada-Instruct paper.

    Note:
        The Genstruct prompt i.e. the task, can be used with any model really, but the safest / recommended
        option is to use `NousResearch/Genstruct-7B` as the LLM provided to the task, since it was trained
        for this specific task.

    Attributes:
        system_prompt: The system prompt for the instruction generation task.
        _template: a Jinja2 template used to format the input for the LLM.

    Input columns:
        - title (`str`): The title of the document.
        - content (`str`): The content of the document.
        - system_prompt (`Optional[str]`): The system prompt for the instruction generation task.

    Output columns:
        - user (`str`): The user's instruction based on the document.
        - assistant (`str`): The assistant's response based on the user's instruction.
        - model_name (`str`): The model name used to generate the `feedback` and `result`.

    Categories:
        - text-generation
        - instruction
        - response

    References:
        - [Genstruct 7B by Nous Research](https://huggingface.co/NousResearch/Genstruct-7B)
        - [Ada-Instruct: Adapting Instruction Generators for Complex Reasoning](https://arxiv.org/abs/2310.04484)

    Examples:
        Generate instructions from raw documents using the title and content:

        ```python
        from distilabel.steps.tasks import Genstruct
        from distilabel.models import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        genstruct = Genstruct(
            llm=InferenceEndpointsLLM(
                model_id="NousResearch/Genstruct-7B",
            ),
        )

        genstruct.load()

        result = next(
            genstruct.process(
                [
                    {"title": "common instruction", "content": "content of the document"},
                ]
            )
        )
        # result
        # [
        #     {
        #         'title': 'An instruction',
        #         'content': 'content of the document',
        #         'model_name': 'test',
        #         'user': 'An instruction',
        #         'assistant': 'content of the document',
        #     }
        # ]
        ```

    Citations:
        ```
        @misc{cui2023adainstructadaptinginstructiongenerators,
            title={Ada-Instruct: Adapting Instruction Generators for Complex Reasoning},
            author={Wanyun Cui and Qianle Wang},
            year={2023},
            eprint={2310.04484},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2310.04484},
        }
        ```
    """

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "genstruct.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        """The inputs for the task are the `title` and the `content`."""
        return {"title": True, "content": True, "system_prompt": False}

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        messages = []
        if "system_prompt" in input:
            messages.append({"role": "system", "content": input["system_prompt"]})
        elif self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    title=input["title"], content=input["content"]
                ),
            }
        )
        return messages

    @property
    def outputs(self) -> List[str]:
        """The output for the task are the `user` instruction based on the provided document
        and the `assistant` response based on the user's instruction."""
        return ["user", "assistant", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted so that both the user and the assistant messages are
        captured.

        Args:
            output: the raw output of the LLM.
            input: the input to the task. Used for obtaining the number of responses.

        Returns:
            A dict with the keys `user` and `assistant` containing the content for each role.
        """
        if output is None:
            return {"user": None, "assistant": None}

        matches = re.search(_PARSE_GENSTRUCT_OUTPUT_REGEX, output, re.DOTALL)
        if not matches:
            return {"user": None, "assistant": None}

        return {
            "user": matches.group(1).strip(),
            "assistant": matches.group(2).strip(),
        }
