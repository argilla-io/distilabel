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

from pydantic import BaseModel

from distilabel.mixins.runtime_parameters import RuntimeParametersMixin
from distilabel.utils.serialization import _Serializable


class KnowledgeBase(RuntimeParametersMixin, BaseModel, _Serializable, ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    def keyword_search(
        self, keywords: List[str], n_retrieved_documents: int
    ) -> List[str]:
        raise NotImplementedError

    def vector_search(
        self, vector: List[float], n_retrieved_documents: int
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def hybrid_search(
        self, vector: List[float], keywords: List[str], n_retrieved_documents: int
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def search(
        self,
        vector: Optional[List[float]] = None,
        keywords: Optional[List[str]] = None,
        n_retrieved_documents: int = 5,
    ) -> List[Dict[str, Any]]:
        if vector is not None and keywords is not None:
            return self.hybrid_search(vector, keywords, n_retrieved_documents)
        elif vector is not None:
            return self.vector_search(vector, n_retrieved_documents)
        elif keywords is not None:
            return self.keyword_search(keywords, n_retrieved_documents)
        else:
            raise ValueError("Either vector or keywords must be provided.")

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        pass
