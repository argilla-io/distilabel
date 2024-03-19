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

import pytest
from datasets import Dataset, DatasetDict
from distilabel.utils.data import Distiset


@pytest.fixture(scope="function")
def distiset():
    return Distiset(
        {
            "leaf_step_1": Dataset.from_dict({"a": [1, 2, 3]}),
            "leaf_step_2": Dataset.from_dict({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}),
        }
    )


class TestDistiset:
    def test_train_test_split(self, distiset):
        assert isinstance(distiset["leaf_step_1"], Dataset)
        ds = distiset.train_test_split(0.8)
        assert isinstance(ds, Distiset)
        assert len(ds) == 2
        assert isinstance(ds["leaf_step_1"], DatasetDict)
