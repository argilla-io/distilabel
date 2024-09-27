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


def script_executed_in_ray_cluster() -> bool:
    """Checks if running in a Ray cluster. The checking is based on the presence of
    typical Ray environment variables that are set in each node of the cluster.

    Returns:
        `True` if running on a Ray cluster, `False` otherwise.
    """
    return all(
        env in os.environ
        for env in ["RAY_NODE_TYPE_NAME", "RAY_CLUSTER_NAME", "RAY_ADDRESS"]
    )
