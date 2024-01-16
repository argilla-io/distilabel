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

from functools import partial, wraps
from typing import Any, Callable, Tuple, TypeVar, Union

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import ParamSpec

_pipeline_progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    MofNCompleteColumn(),
    TimeRemainingColumn(elapsed_when_finished=True),
)

P = ParamSpec("P")
R = TypeVar("R")


def use_progress_bar(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        _pipeline_progress.start()
        try:
            r = func(*args, **kwargs)
        finally:
            _pipeline_progress.stop()
        return r

    return wrapper


def get_progress_bar(*args: Any, **kwargs: Any) -> Callable[..., None]:
    task_id = _pipeline_progress.add_task(*args, **kwargs)

    def update_progress_bar(**kwargs: Any) -> None:
        _pipeline_progress.update(task_id, **kwargs)

    return update_progress_bar


ProgressFunc = Union[Callable[[], None], None]


def get_progress_bars_for_pipeline(
    num_rows: int,
    num_generations: int,
    display_progress_bar: bool,
    has_generator: bool,
    has_labeller: bool,
) -> Tuple[ProgressFunc, ProgressFunc]:
    if display_progress_bar:
        generation_progress_bar = get_progress_bar(
            description="Texts Generated", total=num_rows * num_generations
        )

        def _generation_progress_func(has_generator: bool, advance=None) -> None:
            # If there's no generator, we are not showing the progress bar.
            # This information comes from pipelines.py
            return (
                generation_progress_bar(advance=advance or num_generations)
                if has_generator
                else None
            )

        labelling_progress_bar = get_progress_bar(
            description="Rows labelled", total=num_rows
        )

        def _labelling_progress_func(has_labeller: bool, advance=None) -> None:
            # If there's no labeller, we are not showing the progress bar.
            # This information comes from pipelines.py
            return (
                labelling_progress_bar(advance=advance or 1) if has_labeller else None
            )

        _partial_generation_progress_func = partial(
            _generation_progress_func, has_generator=has_generator
        )
        _partial_labelling_progress_func = partial(
            _labelling_progress_func, has_labeller=has_labeller
        )
        return _partial_generation_progress_func, _partial_labelling_progress_func

    return None, None
