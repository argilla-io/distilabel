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

import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar

T = TypeVar("T")

TASK_FILE_NAME = "task-distilabel.json"


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
    """Writes a json file to the given path, creates the parent dir.

    Args:
        filename (Path): Name of the file.
        data (Dict[str, Any]): Dict to be written as json.
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def read_json(filename: Path) -> Dict[str, Any]:
    """Read a json file from disk.

    Args:
        filename (Path): Name of the json file.

    Returns:
        Dict[str, Any]: Dict containing the json data.
    """
    with open(filename, "r") as file:
        return json.load(file)


class _Serializable:
    """Base class for serializable classes.
    It provides the means to serialize and deserialize.

    We currently defien the tasks as dataclasses, in which case we can
    use the `asdict` method to serialize the class, but we may need to review
    this if we decide to remove the dataclasses.
    Other than the default content, the obtained from `dataclasses.asdict`, we
    we store create a __type_info__ variable to store the relevant information
    to load back the class.
    """

    __type_info__: Dict[str, Any] = {}

    def dump(self) -> Dict[str, Any]:
        """Transforms the class into a dict to write to a file.

        Returns:
            Dict[str, Any]: Serializable content of the class.
        """
        _dict = asdict(self)
        _dict["__type_info__"] = {
            "module": type(self).__module__,
            "name": type(self).__name__,
        }
        return _dict

    def save(self, path: Optional[os.PathLike] = None) -> None:
        """Writes to a file the content.

        Args:
            path (Optional[os.PathLike], optional):
                Filename of the task. If a folder is given, will create the task
                inside. If None is given, the file will be created at the current
                working directory. Defaults to None.
        """
        if path is None:
            path = Path.cwd() / TASK_FILE_NAME
        else:
            path = Path(path)
            if path.suffix == "":
                # If the path has no suffix, assume the user just wants a folder to write the task
                path = path / TASK_FILE_NAME
        write_json(path, self.dump())

    @classmethod
    def from_json(cls, template_path: os.PathLike) -> Generic[T]:
        """Loads a template from a file and returns the instance contained.

        Args:
            template_path (os.PathLike): _description_

        Raises:
            ValueError: _description_

        Returns:
            Generic[T]: _description_
        """
        if Path(template_path).is_dir():
            raise ValueError(
                f"You must provide a file path, not a directory: {template_path}"
            )
        template = read_json(template_path)
        return load_from_dict(template)
