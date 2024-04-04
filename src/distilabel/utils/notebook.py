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


def in_notebook() -> bool:
    """Checks if the current code is being executed from a Jupyter Notebook.
    This is useful for better handling the `asyncio` events under `nest_asyncio`,
    as Jupyter Notebook runs a separate event loop.

    Returns:
        Whether the current code is being executed from a Jupyter Notebook.

    References:
        - https://stackoverflow.com/a/22424821
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
