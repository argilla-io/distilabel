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

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from typing_extensions import override

from distilabel.llm.base import LLM
from distilabel.steps.base import (
    GeneratorStep,
)
from distilabel.steps.task.base import _Task

if TYPE_CHECKING:
    from distilabel.steps.task.typing import ChatType
    from distilabel.steps.typing import GeneratorStepOutput


class DataTask(_Task, GeneratorStep):
    """DataTask is a class that implements the `_Task` abstract class and adds the
    `GeneratorStep` interface to be used as a step in the pipeline.

    Args:
        data: The data to be used to generate the outputs of the task.
    """

    llm: LLM = None
    group_generations: Optional[bool] = None
    num_generations: Optional[int] = None
    generation_kwargs: Optional[dict] = None
    data: List[Dict[str, Any]]

    @override
    def load(self) -> None:
        pass

    @override
    def process(self) -> "GeneratorStepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """
        while self.data:
            batch = self.data[: self.batch_size]
            self.data = self.data[self.batch_size :]
            yield (
                batch,
                True if len(self.data) == 0 else False,
            )

    @property
    def outputs(self) -> List[str]:
        """List of strings with the names of the columns that the step will produce as
        output.

        Returns:
            List of strings with the names of the columns that the step will produce as
            output.
        """
        return list(self.data[0].keys()) + ["model_name"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":  # type: ignore
        pass

    def format_output(self, input: Dict[str, Any]) -> "ChatType":  # type: ignore
        pass
