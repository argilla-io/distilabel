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

import os
from unittest import mock

from distilabel.utils.ray import script_executed_in_ray_cluster


def test_script_executed_on_ray_cluster() -> None:
    assert not script_executed_in_ray_cluster()

    with mock.patch.dict(
        os.environ,
        {
            "RAY_NODE_TYPE_NAME": "headgroup",
            "RAY_CLUSTER_NAME": "disticluster",
            "RAY_ADDRESS": "127.0.0.1:6379",
        },
    ):
        assert script_executed_in_ray_cluster()
