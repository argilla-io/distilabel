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

from unittest import mock

from typer.testing import CliRunner

from distilabel.cli.app import app
from tests.unit.cli.utils import TEST_PIPELINE_PATH

runner = CliRunner()


class TestPipelineRun:
    @mock.patch("distilabel.pipeline.local.Pipeline.run")
    def test_pipeline_run(self, pipeline_run_mock: mock.MagicMock) -> None:
        result = runner.invoke(
            app,
            [
                "pipeline",
                "run",
                "--config",
                TEST_PIPELINE_PATH,
                "--param",
                "load_hub_dataset.repo_id=distilabel-internal-testing/this-does-not-matter-dataset",
                "--param",
                "load_hub_dataset.split=train",
                "--param",
                "text_generation_gpt.num_generations=3",
                "--param",
                "text_generation_gpt_2.num_generations=3",
                "--param",
                "text_generation_gpt_2.llm.generation_kwargs.top_p=0.5",
                "--param",
                "text_generation_gpt_2.llm.generation_kwargs.top_k=10",
                "--param",
                "push_to_hub.repo_id=distilabel-internal-testing/testing-distilabel-push-to-hub",
                "--param",
                "push_to_hub_2.repo_id=distilabel-internal-testing/testing-distilabel-push-to-hub-2",
                "--repo-id=distilabel-internal-testing/testing-distilabel-push-to-hub",
                "--commit-message=Testing",
                "--private",
                "--token",
                "this-is-a-token",
                "--ignore-cache",
            ],
        )

        pipeline_run_mock.assert_called_once_with(
            parameters={
                "load_hub_dataset": {
                    "repo_id": "distilabel-internal-testing/this-does-not-matter-dataset",
                    "split": "train",
                },
                "text_generation_gpt": {"num_generations": "3"},
                "text_generation_gpt_2": {
                    "num_generations": "3",
                    "llm": {"generation_kwargs": {"top_p": "0.5", "top_k": "10"}},
                },
                "push_to_hub": {
                    "repo_id": "distilabel-internal-testing/testing-distilabel-push-to-hub"
                },
                "push_to_hub_2": {
                    "repo_id": "distilabel-internal-testing/testing-distilabel-push-to-hub-2"
                },
            },
            use_cache=False,
        )

        pipeline_run_mock.return_value.push_to_hub.assert_called_once_with(
            repo_id="distilabel-internal-testing/testing-distilabel-push-to-hub",
            commit_message="Testing",
            private=True,
            token="this-is-a-token",
        )

        assert result.exit_code == 0

    @mock.patch("distilabel.pipeline.local.Pipeline.run")
    def test_pipeline_run_without_repo_id(
        self, pipeline_run_mock: mock.MagicMock
    ) -> None:
        result = runner.invoke(
            app,
            ["pipeline", "run", "--config", TEST_PIPELINE_PATH],
        )

        pipeline_run_mock.return_value.push_to_hub.assert_not_called()

        assert result.exit_code == 0

    @mock.patch("distilabel.pipeline.local.Pipeline.run")
    def test_pipeline_run_config_does_not_exist(
        self, pipeline_run_mock: mock.MagicMock
    ) -> None:
        result = runner.invoke(
            app,
            ["pipeline", "run", "--config", "does-not-exist.yaml"],
        )

        assert result.exit_code == 1


class TestPipelineInfo:
    def test_pipeline_info(self) -> None:
        result = runner.invoke(
            app, ["pipeline", "info", "--config", TEST_PIPELINE_PATH]
        )

        assert result.exit_code == 0

    def test_pipeline_run_config_does_not_exist(self) -> None:
        result = runner.invoke(
            app,
            ["pipeline", "info", "--config", "does-not-exist.yaml"],
        )

        assert result.exit_code == 1
