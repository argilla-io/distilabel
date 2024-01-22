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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def evol_instruct_task_as_dict():
    return {
        "__type_info__": {
            "module": "distilabel.tasks.text_generation.evol_instruct",
            "name": "EvolInstructTask",
        },
        "system_prompt": EvolInstructTask().system_prompt,
    }


@pytest.fixture
def judgelm_task_as_dict():
    return {
        "__type_info__": {
            "module": "distilabel.tasks.preference.judgelm",
            "name": "JudgeLMTask",
        },
        "system_prompt": JudgeLMTask().system_prompt,
        "task_description": JudgeLMTask().task_description,
    }


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def ultracm_for_instruction_following_task_as_dict():
    return {
        "__type_info__": {
            "module": "distilabel.tasks.critique.ultracm",
            "name": "UltraCMTask",
        },
        "system_prompt": UltraCMTask().system_prompt,
    }


class TestTextGenerationTaskSerialization:
    def test_dump(self, text_generation_task_as_dict):
        task = TextGenerationTask()
        task_as_dict = task.dump()
        assert all(k in task_as_dict for k in text_generation_task_as_dict)
        assert task_as_dict["__type_info__"] == {
            "module": "distilabel.tasks.text_generation.base",
            "name": "TextGenerationTask",
        }

    def test_load_from_dict(self, text_generation_task_as_dict):
        task: TextGenerationTask = load_from_dict(text_generation_task_as_dict)
        assert isinstance(task, TextGenerationTask)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            task = TextGenerationTask()
            task.save(tmpdirname)
            template_name = Path(tmpdirname) / TASK_FILE_NAME
            assert template_name.exists()
            task_loaded: TextGenerationTask = TextGenerationTask.from_json(
                template_name
            )
            assert isinstance(task_loaded, TextGenerationTask)


class TestSelfInstructTaskSerialization:
    def test_dump(self, self_instruct_task_as_dict):
        task = SelfInstructTask()
        task_as_dict = task.dump()
        assert all(k in task_as_dict for k in self_instruct_task_as_dict)
        assert task_as_dict["__type_info__"] == {
            "module": "distilabel.tasks.text_generation.self_instruct",
            "name": "SelfInstructTask",
        }

    def test_load_from_dict(self, self_instruct_task_as_dict):
        task: SelfInstructTask = load_from_dict(self_instruct_task_as_dict)
        assert isinstance(task, SelfInstructTask)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            task = SelfInstructTask()
            task.save(tmpdirname)
            template_name = Path(tmpdirname) / TASK_FILE_NAME
            assert template_name.exists()
            task_loaded: SelfInstructTask = SelfInstructTask.from_json(template_name)
            assert isinstance(task_loaded, SelfInstructTask)


class TestEvolInstructTaskSerialization:
    def test_dump(self, evol_instruct_task_as_dict):
        task = EvolInstructTask()
        task_as_dict = task.dump()
        assert all(k in task_as_dict for k in evol_instruct_task_as_dict)
        assert task_as_dict["__type_info__"] == {
            "module": "distilabel.tasks.text_generation.evol_instruct",
            "name": "EvolInstructTask",
        }

    def test_load_from_dict(self, evol_instruct_task_as_dict):
        task: EvolInstructTask = load_from_dict(evol_instruct_task_as_dict)
        assert isinstance(task, EvolInstructTask)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            task = EvolInstructTask()
            task.save(tmpdirname)
            template_name = Path(tmpdirname) / TASK_FILE_NAME
            assert template_name.exists()
            task_loaded: EvolInstructTask = EvolInstructTask.from_json(template_name)
            assert isinstance(task_loaded, EvolInstructTask)


class TestJudgeLMTaskSerialization:
    def test_dump(self, judgelm_task_as_dict):
        task = JudgeLMTask()
        task_as_dict = task.dump()
        assert all(k in task_as_dict for k in judgelm_task_as_dict)
        assert task_as_dict["__type_info__"] == {
            "module": "distilabel.tasks.preference.judgelm",
            "name": "JudgeLMTask",
        }

    def test_load_from_dict(self, judgelm_task_as_dict):
        task: JudgeLMTask = load_from_dict(judgelm_task_as_dict)
        assert isinstance(task, JudgeLMTask)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            task = JudgeLMTask()
            task.save(tmpdirname)
            template_name = Path(tmpdirname) / TASK_FILE_NAME
            assert template_name.exists()
            task_loaded: JudgeLMTask = JudgeLMTask.from_json(template_name)
            assert isinstance(task_loaded, JudgeLMTask)


class TestUltraFeedbackTaskSerialization:
    def test_dump(self, ultrafeedback_for_instruction_following_task_as_dict):
        task = UltraFeedbackTask.for_instruction_following()
        task_as_dict = task.dump()
        assert all(
            k in task_as_dict
            for k in ultrafeedback_for_instruction_following_task_as_dict
        )
        assert task_as_dict["__type_info__"] == {
            "module": "distilabel.tasks.preference.ultrafeedback",
            "name": "UltraFeedbackTask",
        }

    def test_load_from_dict(self, ultrafeedback_for_instruction_following_task_as_dict):
        task: UltraFeedbackTask = load_from_dict(
            ultrafeedback_for_instruction_following_task_as_dict
        )
        assert isinstance(task, UltraFeedbackTask)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            task = UltraFeedbackTask.for_instruction_following()
            task.save(tmpdirname)
            template_name = Path(tmpdirname) / TASK_FILE_NAME
            assert template_name.exists()
            task_loaded: UltraFeedbackTask = UltraFeedbackTask.from_json(template_name)
            assert isinstance(task_loaded, UltraFeedbackTask)


class TestUltraJudgeTaskSerialization:
    def test_dump(self, ultrajudge_for_instruction_following_task_as_dict):
        task = UltraJudgeTask()
        task_as_dict = task.dump()
        assert all(
            k in task_as_dict for k in ultrajudge_for_instruction_following_task_as_dict
        )
        assert task_as_dict["__type_info__"] == {
            "module": "distilabel.tasks.preference.ultrajudge",
            "name": "UltraJudgeTask",
        }

    def test_load_from_dict(self, ultrajudge_for_instruction_following_task_as_dict):
        task: UltraJudgeTask = load_from_dict(
            ultrajudge_for_instruction_following_task_as_dict
        )
        assert isinstance(task, UltraJudgeTask)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            task = UltraJudgeTask()
            task.save(tmpdirname)
            template_name = Path(tmpdirname) / TASK_FILE_NAME
            assert template_name.exists()
            task_loaded: UltraJudgeTask = UltraJudgeTask.from_json(template_name)
            assert isinstance(task_loaded, UltraJudgeTask)


class TestPrometheusTaskSerialization:
    def test_dump(self, prometheus_for_instruction_following_task_as_dict):
        task = sample_prompetheus_task
        task_as_dict = task.dump()
        assert all(
            k in task_as_dict for k in prometheus_for_instruction_following_task_as_dict
        )
        assert task_as_dict["__type_info__"] == {
            "module": "distilabel.tasks.critique.prometheus",
            "name": "PrometheusTask",
        }

    def test_load_from_dict(self, prometheus_for_instruction_following_task_as_dict):
        task: PrometheusTask = load_from_dict(
            prometheus_for_instruction_following_task_as_dict
        )
        assert isinstance(task, PrometheusTask)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            task = sample_prompetheus_task
            task.save(tmpdirname)
            template_name = Path(tmpdirname) / TASK_FILE_NAME
            assert template_name.exists()
            task_loaded: PrometheusTask = PrometheusTask.from_json(template_name)
            assert isinstance(task_loaded, PrometheusTask)


class TestUltraCMTaskSerialization:
    def test_dump(self, ultracm_for_instruction_following_task_as_dict):
        task = UltraCMTask()
        task_as_dict = task.dump()
        assert all(
            k in task_as_dict for k in ultracm_for_instruction_following_task_as_dict
        )
        assert task_as_dict["__type_info__"] == {
            "module": "distilabel.tasks.critique.ultracm",
            "name": "UltraCMTask",
        }

    def test_load_from_dict(self, ultracm_for_instruction_following_task_as_dict):
        task: UltraCMTask = load_from_dict(
            ultracm_for_instruction_following_task_as_dict
        )
        assert isinstance(task, UltraCMTask)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            task = UltraCMTask()
            task.save(tmpdirname)
            template_name = Path(tmpdirname) / TASK_FILE_NAME
            assert template_name.exists()
            task_loaded: UltraCMTask = UltraCMTask.from_json(template_name)
            assert isinstance(task_loaded, UltraCMTask)
