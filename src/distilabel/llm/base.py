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

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel, ConfigDict, PrivateAttr

from distilabel.pipeline.logging import get_logger
from distilabel.pipeline.serialization import _Serializable

if TYPE_CHECKING:
    from distilabel.pipeline.step.task.typing import ChatType


class LLM(BaseModel, _Serializable, ABC):
    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),  # type: ignore
    )

    _values: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _logger: logging.Logger = PrivateAttr(get_logger("llm"))

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def generate(
        self, inputs: List["ChatType"], *args: Any, **kwargs: Any
    ) -> List[str]:
        pass
