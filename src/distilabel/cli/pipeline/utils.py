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
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import requests
import yaml
from pydantic import HttpUrl, ValidationError
from pydantic.type_adapter import TypeAdapter

from distilabel.constants import ROUTING_BATCH_FUNCTION_ATTR_NAME, STEP_ATTR_NAME
from distilabel.errors import DistilabelUserError
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


def _download_remote_file(url: str) -> str:
    """Downloads a file from a Hugging Face Hub repository.

    Args:
        url: URL of the file to download.

    Returns:
        The content of the file.
    """
    if "huggingface.co" in url and "HF_TOKEN" in os.environ:
        headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    else:
        headers = None
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response


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
        raise DistilabelUserError(
            f"Unsupported file format for '{url}'. Only JSON and YAML are supported",
            page="sections/how_to_guides/basic/pipeline/?h=seriali#serializing-the-pipeline",
        )
    response = _download_remote_file(url)

    if url.endswith((".yaml", ".yml")):
        content = response.content.decode("utf-8")
        return yaml.safe_load(content)

    return response.json()


def get_pipeline_from_url(url: str, pipeline_name: str = "pipeline") -> "BasePipeline":
    """Downloads the file to the current working directory and loads the pipeline object
    from a python script.

    Args:
        url: The URL pointing to the python script with the pipeline definition.
        pipeline_name: The name of the pipeline in the script.
            I.e: `with Pipeline(...) as pipeline:...`.

    Returns:
        The pipeline instantiated.

    Raises:
        ValueError: If the file format is not supported.
    """
    if not url.endswith(".py"):
        raise DistilabelUserError(
            f"Unsupported file format for '{url}'. It must be a python file.",
            page="sections/how_to_guides/advanced/cli/#distilabel-pipeline-run",
        )
    response = _download_remote_file(url)

    content = response.content.decode("utf-8")
    script_local = Path.cwd() / Path(url).name
    script_local.write_text(content)

    # Add the current working directory to sys.path
    sys.path.insert(0, os.getcwd())
    module = importlib.import_module(str(Path(url).stem))
    pipeline = getattr(module, pipeline_name, None)
    if not pipeline:
        raise ImportError(
            f"The script must contain an object with the pipeline named: '{pipeline_name}' that can be imported"
        )

    return pipeline


def get_pipeline(
    config_or_script: str, pipeline_name: str = "pipeline"
) -> "BasePipeline":
    """Get a pipeline from a configuration file or a remote python script.

    Args:
        config_or_script: The path or URL to the pipeline configuration file
            or URL to a python script.
        pipeline_name: The name of the pipeline in the script.
            I.e: `with Pipeline(...) as pipeline:...`.

    Returns:
        The pipeline.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the configuration file does not exist.
    """
    config = script = None
    if config_or_script.endswith((".json", ".yaml", ".yml")):
        config = config_or_script
    elif config_or_script.endswith(".py"):
        script = config_or_script
    else:
        raise DistilabelUserError(
            "The file must be a valid config file or python script with a pipeline.",
            page="sections/how_to_guides/advanced/cli/#distilabel-pipeline-run",
        )

    if valid_http_url(config_or_script):
        if config:
            data = get_config_from_url(config)
            return Pipeline.from_dict(data)
        return get_pipeline_from_url(script, pipeline_name=pipeline_name)

    if not config:
        raise ValueError(
            f"To run a pipeline from a python script, run it as `python {script}`"
        )

    if Path(config).is_file():
        return Pipeline.from_file(config)

    raise FileNotFoundError(f"File '{config_or_script}' does not exist.")


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

    if any(
        pipeline.dag.get_step(step).get(ROUTING_BATCH_FUNCTION_ATTR_NAME) is not None
        for step in pipeline.dag.G.nodes
    ):
        information.extend(
            [
                "\n",
                _build_routing_batch_function_panel(pipeline),
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
        table: Table,
        runtime_params: List[Dict[str, Any]],
        prefix: str = "",
    ) -> None:
        for param in runtime_params:
            if isinstance(param, str):
                _add_rows(table, runtime_params[param], f"{prefix}{param}.")
                continue

            # nested (for example `LLM` in `Task`)
            if "runtime_parameters_info" in param:
                _add_rows(
                    table=table,
                    runtime_params=param["runtime_parameters_info"],
                    prefix=f"{prefix}{param['name']}.",
                )
            # `LLM` special case
            elif "keys" in param:
                _add_rows(
                    table=table,
                    runtime_params=param["keys"],
                    prefix=f"{prefix}{param['name']}.",
                )
                return
            else:
                optional = param.get("optional", "")
                if optional != "":
                    optional = "Yes" if optional else "No"

                table.add_row(
                    prefix + param["name"], param.get("description"), optional
                )

    steps = []
    for step_name, runtime_params in pipeline.get_runtime_parameters_info().items():
        step = pipeline.dag.get_step(step_name)[STEP_ATTR_NAME]
        class_name = step.__class__.__name__

        table = Table(
            title=f"{step.name} ([bold][magenta]{class_name}[/bold][/magenta])",
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )

        table.add_column("Runtime parameter", style="dim", width=60)
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


def _build_routing_batch_function_panel(pipeline: "BasePipeline") -> "Panel":
    """Builds a panel to display the routing batch function of the pipeline.

    Args:
        pipeline: The pipeline

    Returns:
        A `rich.panel.Panel` containing the routing batch function of the pipeline.
    """
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Step", style="dim", width=18)
    table.add_column("Function", style="dim")
    table.add_column("Description", width=90)

    G = pipeline.dag.G

    for step_name in G.nodes:
        node = pipeline.dag.get_step(step_name)
        if routing_batch_function := node.get(ROUTING_BATCH_FUNCTION_ATTR_NAME):
            table.add_row(
                step_name,
                routing_batch_function.routing_function.__name__,
                routing_batch_function.description,
            )
            continue

    return Panel(
        table,
        title="[magenta]Routing Batch Function[/magenta]",
        style="light_cyan3",
        expand=True,
    )
