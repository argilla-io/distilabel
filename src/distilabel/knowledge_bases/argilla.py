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
from typing import Any, Dict, List

import pkg_resources
from argilla import Query, Similar
from pydantic import Field

from distilabel.knowledge_bases.base import KnowledgeBase
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.utils.argilla import ArgillaBase


class ArgillaKnowledgeBase(KnowledgeBase, ArgillaBase):
    vector_field: RuntimeParameter[str] = Field(
        None, description="The name of the field containing the vector."
    )

    def load(self) -> None:
        ArgillaBase.load(self)

        if pkg_resources.get_distribution("argilla").version < "2.3.0":
            raise ValueError(
                "Argilla version must be 2.3.0 or higher to use ArgillaKnowledgeBase."
            )

        self._dataset = self._client.datasets(
            name=self.dataset, workspace=self.workspace
        )

    def unload(self) -> None:
        self._client = None
        self._dataset = None

    def query(
        self, query_vector: List[float], n_retrieved_documents: int
    ) -> List[Dict[str, Any]]:
        return self._dataset.records(
            query=Query(similar=Similar(name=self.vector_field, vector=query_vector)),
            limit=n_retrieved_documents,
        ).to_list(flatten=True)

    @property
    def columns(self) -> List[str]:
        return list(self._dataset.records(limit=1).to_list(flatten=True)[0].keys())
