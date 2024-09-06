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

app = typer.Typer(help="Commands to run Distilabel UIs.", no_args_is_help=True)


@app.command(name="prompt-checker", help="Run the Distilabel prompt checker UI.")
def run() -> None:
    from distilabel.ui.prompt_checker.app import PromptChecker

    PromptChecker().run()
