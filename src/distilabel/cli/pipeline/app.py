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

import re
from typing import Any, List, Optional, Tuple

import typer
from typing_extensions import Annotated

RUNTIME_PARAM_REGEX = re.compile(r"(?P<key>[^.]+(?:\.[^=]+)+)=(?P<value>.+)")

app = typer.Typer(help="Commands to run and inspect Distilabel pipelines.")

ConfigOption = Annotated[
    str, typer.Option(help="Path or URL to the Distilabel pipeline configuration file.")
]


def parse_runtime_param(value: str) -> Tuple[List[str], str]:
    match = RUNTIME_PARAM_REGEX.match(value)
    if not match:
        raise typer.BadParameter(
            "Runtime parameters must be in the format `key.subkey=value` or"
            " `key.subkey.subsubkey=value`"
        )
    return match.group("key").split("."), match.group("value")


@app.command(name="run", help="Run a Distilabel pipeline.")
def run(
    # `param` is `List[Tuple[Tuple[str, ...], str]]` after parsing
    param: Annotated[
        List[Any],
        typer.Option(help="", parser=parse_runtime_param, default_factory=list),
    ],
    config: Optional[str] = typer.Option(
        None, help="Path or URL to the distilabel pipeline configuration file."
    ),
    script: Optional[str] = typer.Option(
        None,
        help="URL pointing to a python script containing a distilabel pipeline.",
    ),
    pipeline_variable_name: str = typer.Option(
        default="pipeline",
        help="Name of the pipeline in a script. I.e. the 'pipeline' variable in `with Pipeline(...) as pipeline:...`.",
    ),
    ignore_cache: bool = typer.Option(
        False, help="Whether to ignore the cache and re-run the pipeline from scratch."
    ),
    repo_id: str = typer.Option(
        None,
        help="The Hugging Face Hub repository ID to push the resulting dataset to.",
    ),
    commit_message: str = typer.Option(
        None, help="The commit message to use when pushing the dataset."
    ),
    private: bool = typer.Option(
        False, help="Whether to make the resulting dataset private on the Hub."
    ),
    token: str = typer.Option(
        None, help="The Hugging Face Hub API token to use when pushing the dataset."
    ),
) -> None:
    from distilabel.cli.pipeline.utils import get_pipeline, parse_runtime_parameters

    if script:
        if config:
            typer.secho(
                "Only one of `--config` or `--script` can be informed.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)
        do_run = typer.prompt("This will run a remote script, are you sure? (y/n)")
        if do_run.lower() != "y":
            raise typer.Exit(code=0)
    if not config and not script:
        typer.secho(
            "`--config` or `--script` must be informed.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    try:
        pipeline = get_pipeline(config or script, pipeline_name=pipeline_variable_name)
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1) from e

    parameters = parse_runtime_parameters(param)
    distiset = pipeline.run(parameters=parameters, use_cache=not ignore_cache)

    if repo_id is not None:
        distiset.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            token=token,
        )


@app.command(name="info", help="Get information about a Distilabel pipeline.")
def info(config: ConfigOption) -> None:
    from distilabel.cli.pipeline.utils import display_pipeline_information, get_pipeline

    try:
        pipeline = get_pipeline(config)
        display_pipeline_information(pipeline)
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1) from e
