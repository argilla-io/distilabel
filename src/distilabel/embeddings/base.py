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
from typing import List, Union

from pydantic import BaseModel, ConfigDict, PrivateAttr

from distilabel.mixins.runtime_parameters import RuntimeParametersMixin
from distilabel.utils.serialization import _Serializable


class Embeddings(RuntimeParametersMixin, BaseModel, _Serializable, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
        validate_default=True,
        validate_assignment=True,
        extra="forbid",
    )
    _logger: Union[logging.Logger, None] = PrivateAttr(...)

    def load(self) -> None:
        """Method to be called to initialize the `Embeddings`"""
        self._logger = logging.getLogger(f"distilabel.llm.{self.model_name}")

    def unload(self) -> None:
        """Method to be called to unload the `Embeddings` and release any resources."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the model name used for the `Embeddings`."""
        pass

    @abstractmethod
    def encode(self, inputs: List[str]) -> List[List[Union[int, float]]]:
        pass
