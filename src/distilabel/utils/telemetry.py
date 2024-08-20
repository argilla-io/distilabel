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

import logging
import os

_LOGGER = logging.getLogger(__name__)


def is_running_on_docker_container() -> bool:
    """Returns True if the current process is running inside a Docker container, False otherwise."""
    global _in_docker_container

    if _in_docker_container is None:
        _in_docker_container = _has_docker_env() or _has_docker_cgroup()

    return _in_docker_container


def _has_docker_env() -> bool:
    try:
        return os.path.exists("/.dockerenv")
    except Exception as e:
        _LOGGER.warning(f"Error while checking if running in Docker: {e}")
        return False


def _has_docker_cgroup() -> bool:
    try:
        cgroup_path = "/proc/self/cgroup"
        return (
            os.path.exists(cgroup_path)
            and os.path.isfile(cgroup_path)
            and any("docker" in line for line in open(cgroup_path))
        )
    except Exception as e:
        _LOGGER.warning(f"Error while checking if running in Docker: {e}")
        return False


# Private global variables section
_in_docker_container = None
