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

from typing import Any, Callable, Tuple, Union

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
)

_pipeline_progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    MofNCompleteColumn(),
)


def get_progress_bar(*args: Any, **kwargs: Any):
    if not _pipeline_progress.live.is_started:
        _pipeline_progress.start()

    task_id = _pipeline_progress.add_task(*args, **kwargs)

    def update_progress_bar(**kwargs):
        _pipeline_progress.update(task_id, **kwargs)

    return update_progress_bar


ProgressFunc = Union[Callable[[], None], None]


def get_progress_bars_for_pipeline(
    num_rows: int, num_generations: int, display_progress_bar: bool
) -> Tuple[ProgressFunc, ProgressFunc]:
    if display_progress_bar:
        generation_progress_bar = get_progress_bar(
            description="Texts Generated", total=num_rows * num_generations
        )

        def _generation_progress_func() -> None:
            generation_progress_bar(advance=num_generations)

        labelling_progress_bar = get_progress_bar(
            description="Rows labelled", total=num_rows
        )

        def _labelling_progress_func() -> None:
            labelling_progress_bar(advance=1)

        return _generation_progress_func, _labelling_progress_func

    return None, None
