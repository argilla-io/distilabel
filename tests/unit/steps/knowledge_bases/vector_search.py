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

from unittest.mock import Mock, patch

import pytest

from distilabel.embeddings.base import Embeddings
from distilabel.knowledge_bases.lancedb import LanceDBKnowledgeBase
from distilabel.steps.knowledge_bases.vector_search import VectorSearch


@pytest.fixture
def mock_lancedb_kb():
    kb = Mock(spec=LanceDBKnowledgeBase)
    kb.search.return_value = [
        {"id": 1, "text": "Sample text 1", "embedding": [0.1, 0.2, 0.3]},
        {"id": 2, "text": "Sample text 2", "embedding": [0.4, 0.5, 0.6]},
    ]
    kb.columns = ["id", "text", "embedding"]
    return kb


@pytest.fixture
def mock_embeddings():
    embeddings = Mock(spec=Embeddings)
    embeddings.encode.return_value = [[0.1, 0.2, 0.3]]
    return embeddings


def test_vector_search_with_lancedb_and_embeddings(mock_lancedb_kb, mock_embeddings):
    step = VectorSearch(knowledge_base=mock_lancedb_kb, embeddings=mock_embeddings)
    input_data = [{"text": "Query text"}]

    result = next(step.process(input_data))

    mock_embeddings.encode.assert_called_once_with(inputs=["Query text"])
    mock_lancedb_kb.search.assert_called_once_with(
        vector=[0.1, 0.2, 0.3], n_retrieved_documents=5
    )

    assert len(result) == 1
    assert "id" in result[0]
    assert "text" in result[0]
    assert "embedding" in result[0]


def test_vector_search_with_lancedb_only(mock_lancedb_kb):
    step = VectorSearch(knowledge_base=mock_lancedb_kb)
    input_data = [{"embedding": [0.1, 0.2, 0.3]}]

    result = next(step.process(input_data))

    mock_lancedb_kb.search.assert_called_once_with(
        vector=[0.1, 0.2, 0.3], n_retrieved_documents=5
    )

    assert len(result) == 1
    assert "id" in result[0]
    assert "text" in result[0]
    assert "embedding" in result[0]


def test_vector_search_with_invalid_input(mock_lancedb_kb):
    step = VectorSearch(knowledge_base=mock_lancedb_kb)
    input_data = [{"invalid_key": "Invalid input"}]

    with pytest.raises(KeyError):
        next(step.process(input_data))


@patch("distilabel.steps.knowledge_bases.vector_search.group_dicts")
def test_vector_search_group_dicts_called(mock_group_dicts, mock_lancedb_kb):
    step = VectorSearch(knowledge_base=mock_lancedb_kb)
    input_data = [{"embedding": [0.1, 0.2, 0.3]}]

    next(step.process(input_data))

    mock_group_dicts.assert_called_once()


def test_vector_search_custom_n_retrieved_documents(mock_lancedb_kb):
    step = VectorSearch(knowledge_base=mock_lancedb_kb, n_retrieved_documents=10)
    input_data = [{"embedding": [0.1, 0.2, 0.3]}]

    next(step.process(input_data))

    mock_lancedb_kb.search.assert_called_once_with(
        vector=[0.1, 0.2, 0.3], n_retrieved_documents=10
    )
