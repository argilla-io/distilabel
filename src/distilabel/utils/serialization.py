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

# import sys

# if sys.version_info < (3, 9):
#     import importlib_resources
# else:
#     import importlib.resources as importlib_resources
import importlib
import json
import os
from dataclasses import asdict, field
from pathlib import Path
from typing import Any, Dict, Generic, TypeVar

T = TypeVar("T")

TASK_FILE_NAME = "task.json"


def load_from_dict(template: Dict[str, Any]) -> Generic[T]:
    """Reads a template (a class serialized) and returns a the instance
    contained.

    Args:
        template (Dict[str, Any]): Dict containing the template, the dict serialized.

    Returns:
        Generic[T]: Instance contained in the template
    """
    type_info = template.pop("__type_info__")
    mod = importlib.import_module(type_info["module"])
    cls = getattr(mod, type_info["name"])
    instance = cls(**template)
    return instance


def write_json(filename: Path, data: Dict[str, Any]) -> None:
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def read_json(filename: Path) -> Dict[str, Any]:
    with open(filename, "r") as file:
        return json.load(file)


class Serializable:
    """Base class for serializable classes.
    It provides the means to serialize and deserialize.
    """

    # TODO: If we aren't using dataclasses we can remove this
    __type_info__: Dict[str, Any] = field(
        default_factory=dict, repr=False
    )  # Store module and function name

    def dump(self) -> Dict[str, Any]:
        """Transforms the class into a dict to write to a file.

        Returns:
            Dict[str, Any]: _description_
        """
        _dict = asdict(self)
        _dict["__type_info__"] = {
            "module": type(self).__module__,
            "name": type(self).__name__,
        }
        return _dict

    def save(self, path: os.PathLike) -> None:
        """Writes to a file the content."""
        path = Path(path)
        if path.is_dir():
            path = path / TASK_FILE_NAME
        write_json(path, self.dump())

    @classmethod
    def from_json(cls, template_path: os.PathLike) -> Generic[T]:
        """Loads a template from a file and returns the instance contained."""
        if Path(template_path).is_dir():
            raise ValueError(
                f"You must provide a file path, not a directory: {template_path}"
            )
        template = read_json(template_path)
        return load_from_dict(template)
