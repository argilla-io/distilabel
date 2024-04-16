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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import requests
import yaml
from pydantic import HttpUrl, ValidationError
from pydantic.type_adapter import TypeAdapter

from distilabel.pipeline.local import Pipeline

if TYPE_CHECKING:
    from rich.panel import Panel

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
    if "huggingface.co" in url and "HF_TOKEN" in os.environ:
        headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    else:
        headers = None
    response = requests.get(url, headers=headers)
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


def display_pipeline_information(pipeline: "BasePipeline") -> None:
    """Displays the pipeline information to the console.

    Args:
        pipeline: The pipeline.
    """
    from rich.console import Console

    Console().print(_build_pipeline_panel(pipeline))


def _build_pipeline_panel(pipeline: "BasePipeline") -> "Panel":
    """Builds a panel to display the information of the pipeline.

    Args:
        pipeline: The pipeline

    Returns:
        A `rich.panel.Panel` containing the information of the pipeline.
    """
    from rich.console import Group
    from rich.panel import Panel

    information: List[Any] = [f"[bold][magenta]Name:[/bold][/magenta] {pipeline.name}"]

    if pipeline.description:
        information.append(
            f"[bold][magenta]Description:[/bold][/magenta] {pipeline.description}"
        )

    information.extend(
        [
            "\n",
            _build_steps_panel(pipeline),
            "\n",
            _build_steps_connection_panel(pipeline),
        ]
    )

    return Panel(
        Group(*information),
        title="[magenta]Pipeline Information[/magenta]",
        expand=False,
        style="light_cyan3",
    )


def _build_steps_panel(pipeline: "BasePipeline") -> "Panel":
    """Builds a panel to display the information of the steps.

    Args:
        pipeline: The pipeline

    Returns:
        A `rich.panel.Panel` containing the information of the steps.
    """
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table

    def _add_rows(
        table: Table, runtime_params: List[Dict[str, Any]], prefix: str = ""
    ) -> None:
        for param in runtime_params:
            # nested (for example `LLM` in `Task`)
            if "runtime_parameters_info" in param:
                _add_rows(
                    table=table,
                    runtime_params=param["runtime_parameters_info"],
                    prefix=f"{prefix}{param['name']}.",
                )
                continue

            # `LLM` special case
            if "keys" in param:
                _add_rows(
                    table=table,
                    runtime_params=param["keys"],
                    prefix=f"{prefix}{param['name']}.",
                )
                continue

            optional = param.get("optional", "")
            if optional != "":
                optional = "Yes" if optional else "No"

            table.add_row(prefix + param["name"], param.get("description"), optional)

    steps = []
    for step_name, runtime_params in pipeline.get_runtime_parameters_info().items():
        step = pipeline.dag.get_step(step_name)["step"]
        class_name = step.__class__.__name__

        table = Table(
            title=f"{step.name} ([bold][magenta]{class_name}[/bold][/magenta])",
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )

        table.add_column("Runtime parameter", style="dim", width=50)
        table.add_column("Description", width=100)
        table.add_column("Optional", justify="right")
        _add_rows(table, runtime_params)

        steps.append(table)

    return Panel(
        Group(*steps),
        title="[magenta]Steps[/magenta]",
        expand=False,
        padding=(1, 1, 0, 1),
        style="light_cyan3",
    )


def _build_steps_connection_panel(pipeline: "BasePipeline") -> "Panel":
    """Builds a panel to display the connections of the steps of the pipeline.

    Args:
        pipeline: The pipeline

    Returns:
        A `rich.panel.Panel` containing the connection of the steps.
    """
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("From step", style="dim", width=18)
    table.add_column("To steps", style="dim")

    G = pipeline.dag.G

    for node in G.nodes:
        if successors := list(G.successors(node)):
            # Convert list of successors to string
            successors_str = ", ".join(map(str, successors))
            table.add_row(str(node), successors_str)
            continue

        # If a node has no successors, indicate it as such
        table.add_row(str(node), "No downstream steps")

    return Panel(
        table,
        title="[magenta]Steps connections[/magenta]",
        style="light_cyan3",
        expand=True,
    )
