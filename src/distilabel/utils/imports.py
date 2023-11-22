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

import warnings
from typing import List, Union

import pkg_resources


def _check_package_is_available(
    name: str,
    min_version: Union[str, None] = None,
    greater_or_equal: bool = False,
    max_version: Union[str, None] = None,
    lower_or_equal: bool = False,
    excluded_versions: Union[List[str], None] = None,
) -> bool:
    """Checks whether the provided Python package is installed / exists, and also checks that
    its a valid version if the version identifiers are provided."""
    try:
        version = pkg_resources.get_distribution(name).version
        if min_version is not None:
            if (greater_or_equal and version < min_version) or (
                not greater_or_equal and version <= min_version
            ):
                warnings.warn(
                    f"`{name}` is installed, but the installed version is {version}, while "
                    f"the minimum required version for `{name}` is {min_version}. If you are "
                    f"willing to use `distilabel` with `{name}`, please ensure you install it "
                    "from the package extras.",
                    UserWarning,
                    stacklevel=2,
                )
                return False
        if max_version is not None:
            if (lower_or_equal and version > max_version) or (
                not lower_or_equal and version >= max_version
            ):
                warnings.warn(
                    f"`{name}` is installed, but the installed version is {version}, while "
                    f"the maximum allowed version for `{name}` is {max_version}. If you are "
                    f"willing to use `distilabel` with `{name}`, please ensure you install it "
                    "from the package extras.",
                    UserWarning,
                    stacklevel=2,
                )
                return False
        if excluded_versions is not None:
            if version in excluded_versions:
                warnings.warn(
                    f"`{name}` is installed, but the installed version is {version}, which is "
                    "an excluded version because it's not compatible with `distilabel`. If you are "
                    f"willing to use `distilabel` with `{name}`, please ensure you install it "
                    "from the package extras.",
                    UserWarning,
                    stacklevel=2,
                )
                return False
        return True
    except pkg_resources.DistributionNotFound:
        return False


_ARGILLA_AVAILABLE = _check_package_is_available(
    "argilla", min_version="1.16.0", greater_or_equal=True
)
_OPENAI_AVAILABLE = _check_package_is_available(
    "openai", min_version="1.0.0", greater_or_equal=True
)
_LLAMA_CPP_AVAILABLE = _check_package_is_available(
    "llama_cpp", min_version="0.2.0", greater_or_equal=True
)
_VLLM_AVAILABLE = _check_package_is_available(
    "vllm", min_version="0.2.1", greater_or_equal=True
)
_HUGGINGFACE_HUB_AVAILABLE = _check_package_is_available(
    "huggingface_hub", min_version="1.19.0", greater_or_equal=True
)
_TRANSFORMERS_AVAILABLE = _check_package_is_available(
    "transformers", min_version="4.31.1", greater_or_equal=True
) and _check_package_is_available("torch", min_version="2.0.0", greater_or_equal=True)
