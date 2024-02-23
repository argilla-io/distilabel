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
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, PrivateAttr

from distilabel.pipeline.step.task.typing import ChatType


class LLM(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    _values: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def model_name(self) -> str:
        return self._values.get("model_name", None)

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def format_input(self, input: ChatType) -> Any:
        pass

    @abstractmethod
    def generate(self, input: Any, *args: Any, **kwargs: Any) -> str:
        pass
