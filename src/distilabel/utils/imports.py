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

# import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import List, Union

from packaging import version as version_parser


def _check_package_is_available(
    name: str,
    min_version: Union[str, None] = None,
    greater_or_equal: bool = False,
    max_version: Union[str, None] = None,
    lower_or_equal: bool = False,
    excluded_versions: Union[List[str], None] = None,
) -> bool:
    """Checks whether the provided Python package is installed / exists, and also checks that
    its a valid version if the version identifiers are provided.

    Args:
        name (str): Name of the Python package to check.
        min_version (Union[str, None], optional): Minimum required version of the package. Defaults to None.
        greater_or_equal (bool, optional): Whether the installed version must be greater or equal to the minimum required version. Defaults to False.
        max_version (Union[str, None], optional): Maximum allowed version of the package. Defaults to None.
        lower_or_equal (bool, optional): Whether the installed version must be lower or equal to the maximum allowed version. Defaults to False.
        excluded_versions (Union[List[str], None], optional): List of versions that are not compatible with the package. Defaults to None.

    Raises:
        UserWarning: If the package is installed but the version is not compatible with the provided version identifiers.

    Returns:
        bool: Whether the package is installed and the version is compatible with the provided version identifiers.
    """
    try:
        installed_version = version_parser.parse(version(name))
        if min_version is not None:
            min_version = version_parser.parse(min_version)
            if (greater_or_equal and installed_version < min_version) or (
                not greater_or_equal and installed_version <= min_version
            ):
                # warnings.warn(
                #     f"`{name}` is installed, but the installed version is {installed_version}, while "
                #     f"the minimum required version for `{name}` is {min_version}. If you are "
                #     f"willing to use `distilabel` with `{name}`, please ensure you install it "
                #     "from the package extras.",
                #     UserWarning,
                #     stacklevel=2,
                # )
                return False
        if max_version is not None:
            max_version = version_parser.parse(max_version)
            if (lower_or_equal and installed_version > max_version) or (
                not lower_or_equal and installed_version >= max_version
            ):
                # warnings.warn(
                #     f"`{name}` is installed, but the installed version is {installed_version}, while "
                #     f"the maximum allowed version for `{name}` is {max_version}. If you are "
                #     f"willing to use `distilabel` with `{name}`, please ensure you install it "
                #     "from the package extras.",
                #     UserWarning,
                #     stacklevel=2,
                # )
                return False
        if excluded_versions is not None:
            excluded_versions = [version_parser.parse(v) for v in excluded_versions]
            if installed_version in excluded_versions:
                # warnings.warn(
                #     f"`{name}` is installed, but the installed version is {installed_version}, which is "
                #     "an excluded version because it's not compatible with `distilabel`. If you are "
                #     f"willing to use `distilabel` with `{name}`, please ensure you install it "
                #     "from the package extras.",
                #     UserWarning,
                #     stacklevel=2,
                # )
                return False
        return True
    except PackageNotFoundError:
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
    "huggingface_hub", min_version="0.19.0", greater_or_equal=True
)
_TRANSFORMERS_AVAILABLE = _check_package_is_available(
    "transformers", min_version="4.31.1", greater_or_equal=True
) and _check_package_is_available("torch", min_version="2.0.0", greater_or_equal=True)
