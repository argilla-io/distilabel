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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from distilabel.mixins.runtime_parameters import RuntimeParametersMixin
from distilabel.utils.serialization import _Serializable


class KnowledgeBase(RuntimeParametersMixin, BaseModel, _Serializable, ABC):
    context_turn: Optional[int] = Field(
        default=-1,
        description=(
            "The index of the turn to which the context should be added. "
            "-1 means that the context is added to the last turn. "
            "0 means that the context is added to the first turn or system message."
        ),
    )
    context_prompt: Optional[str] = Field(
        default="\nTake the following context into account in your response.\n\n",
        description="The prompt to be used to add the context to the knowledge base.",
    )

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def query(
        self, query_vector: List[float], n_retrieved_documents: int
    ) -> List[Dict[str, Any]]:
        pass

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        pass
