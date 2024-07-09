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

import TYPE_CHECKING
from pydantic import Field

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.base import GeneratorTask

if TYPE_CHECKING:
    from distilabel.steps.typing import GeneratorStepOutput


class MagpieInstructionGenerator(GeneratorTask):
    num_instructions: RuntimeParameter[int] = Field(
        default=None, description="The number of instructions to generate."
    )

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        pass
