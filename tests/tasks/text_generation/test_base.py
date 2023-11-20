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

import pytest
from distilabel.tasks.text_generation.base import TextGenerationTask
from distilabel.tasks.text_generation.principles import UltraFeedbackPrinciples


class TestSuiteTextGenerationTask:
    def test_init(self) -> None:
        task = TextGenerationTask()
        assert task.system_prompt == (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
            " while being safe. Your answers should not include any harmful, unethical, racist, sexist,"
            " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased"
            " and positive in nature.\nIf a question does not make any sense, or is not factually coherent,"
            " explain why instead of answering something not correct. If you don't know the answer to a"
            " question, please don't share false information."
        )
        assert task.principles == {
            "harmlessness": UltraFeedbackPrinciples.harmlessness,
            "helpfulness": UltraFeedbackPrinciples.helpfulness,
            "truthfulness": UltraFeedbackPrinciples.truthfulness,
            "honesty": UltraFeedbackPrinciples.honesty,
            "verbalized_calibration": UltraFeedbackPrinciples.verbalized_calibration,
        }
        assert task.principles_distribution is None

    def test_init_with_missing_principles_distribution(self) -> None:
        with pytest.raises(
            ValueError,
            match="Principles 'helpfulness', 'truthfulness', 'honesty', 'verbalized_calibration' included in `principles` is not in `principles_distribution`",
        ):
            TextGenerationTask(principles_distribution={"harmlessness": 1.0})

    def test_init_with_principles_distribution_not_suming_up_1(self) -> None:
        with pytest.raises(
            ValueError, match="`principles_distribution` must sum to 1.0"
        ):
            TextGenerationTask(
                principles_distribution={
                    "harmlessness": 1.0,
                    "helpfulness": 1.0,
                    "truthfulness": 1.0,
                    "honesty": 1.0,
                    "verbalized_calibration": 1.0,
                }
            )

    def test_generate_prompt(self) -> None:
        task = TextGenerationTask(system_prompt="This is a custom system prompt")
        prompt = task.generate_prompt(input="Generate something my boy")
        assert prompt.system_prompt == "This is a custom system prompt"
        assert prompt.formatted_prompt == "Generate something my boy"

    def test_generate_prompt_with_default(self) -> None:
        task = TextGenerationTask()
        prompt = task.generate_prompt(input="Generate something my boy")
        assert prompt.system_prompt == task.system_prompt
        assert prompt.formatted_prompt == "Generate something my boy"

    def test_generate_prompt_with_balanced_principles(self) -> None:
        task = TextGenerationTask(
            principles={"helpfulness": ["Help! I need somebody..."]},
            principles_distribution="balanced",
        )
        prompt = task.generate_prompt(input="Generate something my boy")
        assert (
            prompt.system_prompt
            == task.system_prompt + " " + "Help! I need somebody..."
        )

    def test_generate_prompt_with_distribution_principles(self) -> None:
        task = TextGenerationTask(
            principles={
                "helpfulness": ["Help! I need somebody..."],
                "honesty": ["I'm honest, I swear!"],
            },
            principles_distribution={"helpfulness": 1.0, "honesty": 0.0},
        )
        prompt = task.generate_prompt(input="Generate something my boy")
        assert (
            prompt.system_prompt
            == task.system_prompt + " " + "Help! I need somebody..."
        )
