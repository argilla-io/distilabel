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


from datetime import timedelta
from typing import Any, Dict, List, Optional

import lancedb
from lancedb import DBConnection
from lancedb.table import Table
from pydantic import Field, PrivateAttr

from distilabel.knowledge_bases.base import KnowledgeBase


class LanceDBKnowledgeBase(KnowledgeBase):
    uri: str = Field(..., description="The URI of the LanceDB database.")
    table_name: str = Field(..., description="The name of the table to use.")
    api_key: Optional[str] = Field(
        None, description="The API key to use to connect to the LanceDB database."
    )
    region: Optional[str] = Field(
        None, description="The region of the LanceDB database."
    )
    read_consistency_interval: Optional[timedelta] = Field(
        None, description="The read consistency interval of the LanceDB database."
    )
    request_thread_pool_size: Optional[int] = Field(
        None, description="The request thread pool size of the LanceDB database."
    )
    index_cache_size: Optional[int] = Field(
        None, description="The index cache size of the LanceDB database."
    )

    _db: DBConnection = PrivateAttr(None)
    _tbl: Table = PrivateAttr(None)

    def load(self) -> None:
        self._db = lancedb.connect(self.uri)
        self._tbl = self._db.open_table(name=self.table_name)

    def unload(self) -> None:
        self._db.close()

    def query(
        self, query_vector: List[float], n_retrieved_documents: int
    ) -> List[Dict[str, Any]]:
        return self._tbl.search(query_vector).limit(n_retrieved_documents).to_list()

    @property
    def columns(self) -> List[str]:
        return self._tbl.schema.names
