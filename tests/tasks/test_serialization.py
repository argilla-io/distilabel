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
from distilabel.tasks.text_generation.base import TextGenerationTask
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
            template_name = Path(tmpdirname, TASK_FILE_NAME)
            assert template_name.exists()
            task_loaded: TextGenerationTask = TextGenerationTask.from_json(
                template_name
            )
            assert isinstance(task_loaded, TextGenerationTask)
