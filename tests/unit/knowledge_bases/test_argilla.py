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

from unittest.mock import Mock

import pytest

from distilabel.errors import DistilabelUserError
from distilabel.knowledge_bases.argilla import ArgillaKnowledgeBase


@pytest.fixture
def kb():
    kb = ArgillaKnowledgeBase(
        dataset_name="test",
        dataset_workspace="test",
        vector_field="embedding",
        api_url="http://localhost:6900",
        api_key="test",
    )

    yield kb


class TestArgillaKnowledgeBase:
    def test_initialization_wo_apikeys(self, kb: ArgillaKnowledgeBase):
        kb.api_url = None
        kb.api_key = None
        with pytest.raises(DistilabelUserError):
            kb.load()

    def test_vector_search(self, kb: ArgillaKnowledgeBase):
        # Mock the Argilla client and dataset
        mock_client = Mock()
        mock_dataset = Mock()
        kb._client = mock_client
        kb._dataset = mock_dataset

        # Mock the records method of the dataset
        mock_records = Mock()
        mock_dataset.records.return_value = mock_records

        # Mock the to_list method of the records
        mock_records.to_list.return_value = [
            {"id": 1, "text": "Sample text 1", "embedding": [0.1, 0.2, 0.3]},
            {"id": 2, "text": "Sample text 2", "embedding": [0.4, 0.5, 0.6]},
        ]

        # Perform vector search
        query_vector = [0.1, 0.2, 0.3]
        results = kb.vector_search(query_vector, n_retrieved_documents=2)

        # Assert that the correct methods were called
        mock_dataset.records.assert_called_once()
        mock_records.to_list.assert_called_once_with(flatten=True)

        # Assert the results
        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2

    def test_columns(self, kb: ArgillaKnowledgeBase):
        # Mock the Argilla client and dataset
        mock_client = Mock()
        mock_dataset = Mock()
        kb._client = mock_client
        kb._dataset = mock_dataset

        # Mock the records method of the dataset
        mock_records = Mock()
        mock_dataset.records.return_value = mock_records

        # Mock the to_list method of the records
        mock_records.to_list.return_value = [
            {"id": 1, "text": "Sample text", "embedding": [0.1, 0.2, 0.3]}
        ]

        # Get columns
        columns = kb.columns

        # Assert that the correct methods were called
        mock_dataset.records.assert_called_once_with(limit=1)
        mock_records.to_list.assert_called_once_with(flatten=True)

        # Assert the columns
        assert set(columns) == {"id", "text", "embedding"}
