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

import inspect
from typing import Generator, List, Type, TypedDict, TypeVar

from distilabel.models.embeddings.base import Embeddings
from distilabel.models.llms.base import LLM
from distilabel.steps.base import _Step
from distilabel.steps.tasks.base import _Task
from distilabel.steps.tasks.generate_embeddings import GenerateEmbeddings
from distilabel.steps.tasks.pair_rm import PairRM
from distilabel.utils.docstring import parse_google_docstring


class ComponentsInfo(TypedDict):
    """A dictionary containing `distilabel` components information."""

    llms: List
    ilms: List
    steps: List
    tasks: List
    embeddings: List


def export_components_info() -> ComponentsInfo:
    """Exports `distilabel` components (`LLM`s, `Step`s and `Task`s) information in a dictionary
    format. This information can be used to generate `distilabel` components documentation,
    or to be used in 3rd party applications (UIs, etc).

    Returns:
        A dictionary containing `distilabel` components information
    """

    return {
        "steps": [
            {"name": step_type.__name__, "docstring": parse_google_docstring(step_type)}
            for step_type in _get_steps()
        ],
        "tasks": [
            {"name": task_type.__name__, "docstring": parse_google_docstring(task_type)}
            for task_type in _get_tasks()
        ],
        "llms": [
            {"name": llm_type.__name__, "docstring": parse_google_docstring(llm_type)}
            for llm_type in _get_llms()
        ],
        "ilms": [
            {"name": ilm_type.__name__, "docstring": parse_google_docstring(ilm_type)}
            for ilm_type in _get_ilms()
        ],
        "embeddings": [
            {
                "name": embeddings_type.__name__,
                "docstring": parse_google_docstring(embeddings_type),
            }
            for embeddings_type in _get_embeddings()
        ],
    }


T = TypeVar("T", covariant=True)


def _get_steps() -> List[Type["_Step"]]:
    """Get all `Step` subclasses, that are not abstract classes and not `Task` subclasses.

    Returns:
        A list of `Step` subclasses, except `Task` subclasses
    """
    return [
        step_type
        for step_type in _recursive_subclasses(_Step)
        if not inspect.isabstract(step_type)
        and not issubclass(step_type, _Task)
        and step_type not in [PairRM, GenerateEmbeddings]
    ]


def _get_tasks() -> List[Type["_Task"]]:
    """Get all `Task` subclasses, that are not abstract classes.

    Returns:
        A list of `Task` subclasses
    """
    tasks = [
        task_type
        for task_type in _recursive_subclasses(_Task)
        if not inspect.isabstract(task_type)
    ]

    tasks.extend([PairRM, GenerateEmbeddings])  # type: ignore

    return tasks


def _get_llms() -> List[Type["LLM"]]:
    """Get all `LLM` subclasses, that are not abstract classes.

    Returns:
        A list of `LLM` subclasses, except `AsyncLLM` subclass
    """
    return [
        llm_type
        for llm_type in _recursive_subclasses(LLM)
        if not inspect.isabstract(llm_type)
    ]


def _get_ilms() -> List[Type["LLM"]]:
    """Get all `ILM` subclasses, that are not abstract classes.

    Note:
        This is a placeholder as we don't have `ILM` classes yet.

    Returns:
        The list of all the classes under `distilabel.models.ilms` that are not abstract classes.
    """
    return [
        llm_type
        for llm_type in _recursive_subclasses(LLM)
        if ("distilabel.models.ilms" in str(llm_type))
        and (not inspect.isabstract(llm_type))
    ]


def _get_embeddings() -> List[Type["Embeddings"]]:
    """Get all `Embeddings` subclasses, that are not abstract classes.

    Returns:
        A list of `Embeddings` subclasses, except `AsyncLLM` subclass
    """
    return [
        embeddings_type
        for embeddings_type in _recursive_subclasses(Embeddings)
        if not inspect.isabstract(embeddings_type)
    ]


# Reference: https://adamj.eu/tech/2024/05/10/python-all-subclasses/
def _recursive_subclasses(klass: Type[T]) -> Generator[Type[T], None, None]:
    """Recursively get all subclasses of a class.

    Args:
        klass: A class to get subclasses from.

    Yield:
        A generator of subclasses of the given class.
    """
    for subclass in klass.__subclasses__():
        yield subclass
        yield from _recursive_subclasses(subclass)
