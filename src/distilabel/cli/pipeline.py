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

import typer
from typing_extensions import Annotated

app = typer.Typer(help="Commands to run and inspect Distilabel pipelines.")

ConfigOption = Annotated[
    str, typer.Option(help="Path or URL to the Distilabel pipeline configuration file.")
]


@app.command(name="run", help="Run a Distilabel pipeline.")
def run(config: ConfigOption) -> None:
    # TODO: check if config is a valid file or URL
    pass


@app.command(name="info", help="Get information about a Distilabel pipeline.")
def info(config: ConfigOption) -> None:
    # TODO: check if config is a valid file or URL
    pass
