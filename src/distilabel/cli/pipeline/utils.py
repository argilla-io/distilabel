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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import requests
import yaml
from pydantic import HttpUrl, ValidationError
from pydantic.type_adapter import TypeAdapter

from distilabel.pipeline.local import Pipeline

if TYPE_CHECKING:
    from distilabel.pipeline.base import BasePipeline


def parse_runtime_parameters(
    params: List[Tuple[List[str], str]],
) -> Dict[str, Dict[str, Any]]:
    """Parses the runtime parameters from the CLI format to the format expected by the
    `Pipeline.run` method. The CLI format is a list of tuples, where the first element is
    a list of keys and the second element is the value.

    Args:
        params: A list of tuples, where the first element is a list of keys and the
            second element is the value.

    Returns:
        A dictionary with the runtime parameters in the format expected by the
        `Pipeline.run` method.
    """
    runtime_params = {}
    for keys, value in params:
        current = runtime_params
        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                current[key] = value
            else:
                current = current.setdefault(key, {})
    return runtime_params


def valid_http_url(url: str) -> bool:
    """Check if the URL is a valid HTTP URL.

    Args:
        url: The URL to check.

    Returns:
        `True`, if the URL is a valid HTTP URL. `False`, otherwise.
    """
    try:
        TypeAdapter(HttpUrl).validate_python(url)  # type: ignore
    except ValidationError:
        return False

    return True


def get_config_from_url(url: str) -> Dict[str, Any]:
    """Loads the pipeline configuration from a URL pointing to a JSON or YAML file.

    Args:
        url: The URL pointing to the pipeline configuration file.

    Returns:
        The pipeline configuration as a dictionary.

    Raises:
        ValueError: If the file format is not supported.
    """
    if not url.endswith((".json", ".yaml", ".yml")):
        raise ValueError(
            f"Unsupported file format for '{url}'. Only JSON and YAML are supported"
        )

    response = requests.get(url)
    response.raise_for_status()

    if url.endswith((".yaml", ".yml")):
        content = response.content.decode("utf-8")
        return yaml.safe_load(content)

    return response.json()


def get_pipeline(config: str) -> "BasePipeline":
    """Get a pipeline from a configuration file.

    Args:
        config: The path or URL to the pipeline configuration file.

    Returns:
        The pipeline.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the configuration file does not exist.
    """
    if valid_http_url(config):
        return Pipeline.from_dict(get_config_from_url(config))

    if Path(config).is_file():
        return Pipeline.from_file(config)

    raise FileNotFoundError(f"Config file '{config}' does not exist.")


def build_markdown(pipeline: "BasePipeline") -> str:
    """Builds a markdown string with the pipeline information.

    Args:
        pipeline: The pipeline.

    Returns:
        The markdown string.
    """
    markdown = "**Pipeline Information**\n\n"
    markdown += f"- **Name**: `{pipeline.name}`\n"
    if pipeline.description:
        markdown += f"- **Description**: `{pipeline.description}`\n\n"

    markdown += "**Steps**\n\n"
    for step_name, runtime_params in pipeline.get_runtime_parameters_info().items():
        step = pipeline.dag.get_step(step_name)["step"]
        class_name = step.__class__.__name__
        markdown += f"- `{step_name}`\n"
        markdown += f"  - *Type*: `{class_name}`\n"
        markdown += "  - *Runtime Parameters*:\n"
        for info in runtime_params:
            name = info["name"]
            description = info["description"]
            optional = info["optional"]
            markdown += f"    - `{name}`: {description}\n"
            markdown += f"      - *Optional*: {optional}\n"

    return markdown


def print_pipeline_info(pipeline: "BasePipeline") -> None:
    """Prints the pipeline information to the console.

    Args:
        pipeline: The pipeline.
    """
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    console.print(Markdown(build_markdown(pipeline)))
