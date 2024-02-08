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

from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List

StepInput = Annotated[List[Dict[str, Any]], "StepInput"]


class Step(ABC):
    @property
    @abstractmethod
    def inputs(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def outputs(self) -> List[str]:
        pass

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        pass

    def is_generator(self) -> bool:
        return isinstance(self, GeneratorStep)

    def is_global(self) -> bool:
        return isinstance(self, GlobalStep)


class GeneratorStep(Step, ABC):
    pass


class GlobalStep(Step, ABC):
    pass
