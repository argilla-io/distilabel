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

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import pytest
from distilabel.tasks.critique.prometheus import PrometheusTask
from distilabel.tasks.critique.ultracm import UltraCMTask
from distilabel.tasks.preference.judgelm import JudgeLMTask
from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask
from distilabel.tasks.preference.ultrajudge import UltraJudgeTask
from distilabel.tasks.text_generation.base import TextGenerationTask
from distilabel.tasks.text_generation.evol_instruct import EvolInstructTask
from distilabel.tasks.text_generation.self_instruct import SelfInstructTask
from distilabel.utils.serialization import TASK_FILE_NAME, load_from_dict

if TYPE_CHECKING:
    from distilabel.tasks import Task


def text_generation_task_as_dict():
    return {
        "__type_info__": {
            "module": "distilabel.tasks.text_generation.base",
            "name": "TextGenerationTask",
        },
        "system_prompt": TextGenerationTask().system_prompt,
        "principles": TextGenerationTask().principles,
        "principles_distribution": TextGenerationTask().principles_distribution,
    }


def self_instruct_task_as_dict():
    return {
        "__type_info__": {
            "module": "distilabel.tasks.text_generation.self_instruct",
            "name": "SelfInstructTask",
        },
        "system_prompt": SelfInstructTask().system_prompt,
        "application_description": SelfInstructTask().application_description,
        "num_instructions": SelfInstructTask().num_instructions,
        "criteria_for_query_generation": SelfInstructTask().criteria_for_query_generation,
    }


def evol_instruct_task_as_dict():
    return {
        "__type_info__": {
            "module": "distilabel.tasks.text_generation.evol_instruct",
            "name": "EvolInstructTask",
        },
        "system_prompt": EvolInstructTask().system_prompt,
    }


def judgelm_task_as_dict():
    return {
        "__type_info__": {
            "module": "distilabel.tasks.preference.judgelm",
            "name": "JudgeLMTask",
        },
        "system_prompt": JudgeLMTask().system_prompt,
        "task_description": JudgeLMTask().task_description,
    }


def ultrafeedback_for_instruction_following_task_as_dict():
    # Just one of the cases should be enough
    return {
        "__type_info__": {
            "module": "distilabel.tasks.preference.ultrafeedback",
            "name": "UltraFeedbackTask",
        },
        "system_prompt": UltraFeedbackTask.for_instruction_following().system_prompt,
        "task_description": UltraFeedbackTask.for_instruction_following().task_description,
        "ratings": UltraFeedbackTask.for_instruction_following().ratings,
    }


def ultrajudge_for_instruction_following_task_as_dict():
    return {
        "__type_info__": {
            "module": "distilabel.tasks.preference.ultrajudge",
            "name": "UltraJudgeTask",
        },
        "system_prompt": UltraJudgeTask().system_prompt,
        "task_description": UltraJudgeTask().task_description,
        "areas": UltraJudgeTask().areas,
    }


sample_prompetheus_task = PrometheusTask(
    scoring_criteria="Overall quality of the responses provided.",
    score_descriptions={
        0: "false",
        1: "partially false",
        2: "average",
        3: "partially true",
        4: "true",
    },
)


def prometheus_for_instruction_following_task_as_dict():
    return {
        "__type_info__": {
            "module": "distilabel.tasks.critique.prometheus",
            "name": "PrometheusTask",
        },
        "system_prompt": sample_prompetheus_task.system_prompt,
        "scoring_criteria": sample_prompetheus_task.scoring_criteria,
        "score_descriptions": sample_prompetheus_task.score_descriptions,
    }


def ultracm_for_instruction_following_task_as_dict():
    return {
        "__type_info__": {
            "module": "distilabel.tasks.critique.ultracm",
            "name": "UltraCMTask",
        },
        "system_prompt": UltraCMTask().system_prompt,
    }


@pytest.mark.parametrize(
    "task_as_dict, task_class, type_info",
    [
        (
            text_generation_task_as_dict(),
            TextGenerationTask,
            {
                "module": "distilabel.tasks.text_generation.base",
                "name": "TextGenerationTask",
            },
        ),
        (
            self_instruct_task_as_dict(),
            SelfInstructTask,
            {
                "module": "distilabel.tasks.text_generation.self_instruct",
                "name": "SelfInstructTask",
            },
        ),
        (
            evol_instruct_task_as_dict(),
            EvolInstructTask,
            {
                "module": "distilabel.tasks.text_generation.evol_instruct",
                "name": "EvolInstructTask",
            },
        ),
        (
            judgelm_task_as_dict(),
            JudgeLMTask,
            {
                "module": "distilabel.tasks.preference.judgelm",
                "name": "JudgeLMTask",
            },
        ),
        (
            ultrafeedback_for_instruction_following_task_as_dict(),
            UltraFeedbackTask,
            {
                "module": "distilabel.tasks.preference.ultrafeedback",
                "name": "UltraFeedbackTask",
            },
        ),
        (
            ultrajudge_for_instruction_following_task_as_dict(),
            UltraJudgeTask,
            {
                "module": "distilabel.tasks.preference.ultrajudge",
                "name": "UltraJudgeTask",
            },
        ),
        (
            prometheus_for_instruction_following_task_as_dict(),
            sample_prompetheus_task,
            {
                "module": "distilabel.tasks.critique.prometheus",
                "name": "PrometheusTask",
            },
        ),
        (
            ultracm_for_instruction_following_task_as_dict(),
            UltraCMTask,
            {
                "module": "distilabel.tasks.critique.ultracm",
                "name": "UltraCMTask",
            },
        ),
    ],
)
class TestTaskSerialization:
    def test_dump(
        self,
        task_as_dict: Dict[str, Any],
        task_class: "Task",
        type_info: Dict[str, Any],
    ):
        if task_class == UltraFeedbackTask:
            task_class = UltraFeedbackTask.for_instruction_following
            task_as_dict = task_class().dump()
        elif isinstance(task_class, PrometheusTask):
            task_class = sample_prompetheus_task
            task_as_dict = sample_prompetheus_task.dump()
        else:
            task_as_dict = task_class().dump()

        assert all(k in task_as_dict for k in task_as_dict)
        assert task_as_dict["__type_info__"] == type_info
        task_as_dict.pop("__type_info__")
        assert all(not k.startswith("__") for k in task_as_dict.keys())

    def test_load_from_dict(
        self,
        task_as_dict: Dict[str, Any],
        task_class: "Task",
        type_info: Dict[str, Any],
    ):
        task: "Task" = load_from_dict(task_as_dict)
        if isinstance(task_class, PrometheusTask):
            task_class = type(task_class)
        assert isinstance(task, task_class)

    def test_save_and_load(
        self,
        task_as_dict: Dict[str, Any],
        task_class: "Task",
        type_info: Dict[str, Any],
    ):
        if task_class == UltraFeedbackTask:
            task_class = UltraFeedbackTask.for_instruction_following
        with tempfile.TemporaryDirectory() as tmpdirname:
            if isinstance(task_class, PrometheusTask):
                task_class = sample_prompetheus_task
                task_class.save(tmpdirname)
            else:
                task_class().save(tmpdirname)

            template_name = Path(tmpdirname) / TASK_FILE_NAME
            assert template_name.exists()
            if task_class == UltraFeedbackTask.for_instruction_following:
                task_class = task_class()
            elif isinstance(task_class, PrometheusTask):
                task_class = type(task_class)
            task_loaded: "Task" = task_class.from_json(template_name)
            if isinstance(task_class, UltraFeedbackTask):
                task_class = UltraFeedbackTask
            assert isinstance(task_loaded, task_class)
