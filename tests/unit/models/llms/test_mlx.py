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
import json
import platform
from typing import Any, Dict, Generator

import pytest
from pydantic import BaseModel

from distilabel.models.llms.mlx import MlxLLM

from .utils import DummyUserDetail

RUNS_ON_APPLE_SILICON = platform.processor() == "arm" and platform.system() == "Darwin"


@pytest.mark.skipif(
    not RUNS_ON_APPLE_SILICON,
    reason="MLX only runs on Apple Silicon",
)
@pytest.fixture(scope="module")
def llm() -> Generator[MlxLLM, None, None]:
    llm = MlxLLM(path_or_hf_repo="mlx-community/Qwen2.5-0.5B-4bit")
    llm.load()
    yield llm


@pytest.mark.skipif(
    not RUNS_ON_APPLE_SILICON,
    reason="MLX only runs on Apple Silicon",
)
class TestMlxLLM:
    def test_model_name(self, llm: MlxLLM) -> None:
        assert llm.path_or_hf_repo == "mlx-community/Qwen2.5-0.5B-4bit"

    def test_generate(self, llm: MlxLLM) -> None:
        responses = llm.generate(
            inputs=[
                [{"role": "user", "content": "Hello, how are you?"}],
                [
                    {
                        "role": "user",
                        "content": "You're GPT2, you're old now but you still serves a purpose which is being used in unit tests.",
                    }
                ],
            ],
            num_generations=3,
        )
        assert len(responses) == 2
        generations = responses[0]["generations"]
        statistics = responses[0]["statistics"]
        assert len(generations) == 3
        assert "input_tokens" in statistics
        assert "output_tokens" in statistics

    def test_structured_generation_json(self, llm: MlxLLM) -> None:

        class User(BaseModel):
            first_name: str
            last_name: str

        llm.structured_output = {
            "format": "json",
            "schema": User.model_json_schema()
        }

        responses = llm.generate(
            inputs=[
                [{"role": "user",
                  "content": "Create a user profile for John Smith"}],
            ],
            num_generations=1
        )

        assert len(responses) == 1
        assert "generations" in responses[0]
        assert "statistics" in responses[0]
        generations = responses[0]["generations"]
        assert len(generations) == 1

        # Clean and parse the generation
        for generation in generations:
            # Remove the <|im_end|> tokens and clean up the string
            cleaned_json = generation.replace('<|im_end|>', '').strip()
            try:
                user_data = json.loads(cleaned_json)
                parsed_user = User(**user_data)
                assert isinstance(parsed_user, User)
                assert parsed_user.first_name == "John"
                assert parsed_user.last_name == "Smith"
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw generation: {cleaned_json}")
                raise
            except ValueError as e:
                print(f"Validation error: {e}")
                raise
        statistics = responses[0]["statistics"]
        assert "input_tokens" in statistics
        assert "output_tokens" in statistics

    @pytest.mark.parametrize(
        "structured_output, dump",
        [
            (
                None,
                {
                    "path_or_hf_repo": "mlx-community/Qwen2.5-0.5B-4bit",
                    "generation_kwargs": {},
                    "structured_output": None,
                    "adapter_path": None,
                    "jobs_ids": None,
                    "offline_batch_generation_block_until_done": None,
                    "use_offline_batch_generation": False,
                    "magpie_pre_query_template": None,
                    "tokenizer_config": {},
                    "use_magpie_template": False,
                    "type_info": {
                        "module": "distilabel.models.llms.mlx",
                        "name": "MlxLLM",
                    },
                },
            ),
            (
                {
                    "schema": DummyUserDetail.model_json_schema(),
                    "format": "json",
                },
                {
                    "path_or_hf_repo": "mlx-community/Qwen2.5-0.5B-4bit",
                    "generation_kwargs": {},
                    "magpie_pre_query_template": None,
                    "tokenizer_config": {},
                    "use_magpie_template": False,
                    "structured_output": {
                        "schema": DummyUserDetail.model_json_schema(),
                        "format": "json",
                    },
                    "adapter_path": None,
                    "jobs_ids": None,
                    "offline_batch_generation_block_until_done": None,
                    "use_offline_batch_generation": False,
                    "type_info": {
                        "module": "distilabel.models.llms.mlx",
                        "name": "MlxLLM",
                    },
                },
            ),
        ],
    )
    def test_serialization(
        self, structured_output: Dict[str, Any], dump: Dict[str, Any]
    ) -> None:
        llm = MlxLLM(
            path_or_hf_repo="mlx-community/Qwen2.5-0.5B-4bit",
            structured_output=structured_output,
        )

        assert llm.dump() == dump
        assert isinstance(MlxLLM.from_dict(dump), MlxLLM)
