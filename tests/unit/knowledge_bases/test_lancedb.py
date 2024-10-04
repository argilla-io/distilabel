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

import shutil

import lancedb
import pytest

from distilabel.knowledge_bases.lancedb import LanceDBKnowledgeBase


@pytest.fixture
def kb():
    uri = "data/sample-lancedb"
    table_name = "my_table"
    db = lancedb.connect(uri)
    data = [
        {"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
        {"vector": [5.9, 26.5], "item": "bar", "price": 20.0},
    ]
    try:
        db.create_table(table_name, data=data)
    except OSError:
        pass
    kb = LanceDBKnowledgeBase(uri=uri, table_name=table_name)
    kb.load()
    yield kb

    db.drop_table(table_name)
    shutil.rmtree(uri, ignore_errors=True)


class TestLanceDBKnowledgeBase:
    def test_initialization(self, kb: LanceDBKnowledgeBase):
        assert kb.uri == "data/sample-lancedb"
        assert kb.table_name == "my_table"

    def test_columns(self, kb: LanceDBKnowledgeBase):
        assert kb.columns == ["vector", "item", "price"]

    def test_vector_search(self, kb: LanceDBKnowledgeBase):
        res = kb.vector_search([3.1, 4.1], 1)
        assert isinstance(res, list)
        assert len(res) == 1
        assert res[0]["item"] == "foo"

    def test_hybrid_search(self, kb: LanceDBKnowledgeBase):
        with pytest.raises(NotImplementedError):
            kb.hybrid_search([3.1, 4.1], ["foo"], 1)

    def test_keyword_search(self, kb: LanceDBKnowledgeBase):
        with pytest.raises(NotImplementedError):
            kb.keyword_search(["foo"], 1)
