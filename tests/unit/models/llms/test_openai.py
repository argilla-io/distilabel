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

import os
import sys
from textwrap import dedent
from typing import Any, Dict, List
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import nest_asyncio
import orjson
import pytest
from openai.types import Batch

from distilabel.exceptions import DistilabelOfflineBatchGenerationNotFinishedException
from distilabel.models.llms.openai import _OPENAI_BATCH_API_MAX_FILE_SIZE, OpenAILLM

from .utils import DummyUserDetail


@patch("openai.OpenAI")
@patch("openai.AsyncOpenAI")
class TestOpenAILLM:
    model_id: str = "gpt-4"

    def test_openai_llm(
        self, _async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore

        assert isinstance(llm, OpenAILLM)
        assert llm.model_name == self.model_id

    def test_openai_llm_env_vars(
        self, _async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        with mock.patch.dict(os.environ, clear=True):
            os.environ["OPENAI_API_KEY"] = "another.api.key"
            os.environ["OPENAI_BASE_URL"] = "https://example.com"

            llm = OpenAILLM(model=self.model_id)

            assert isinstance(llm, OpenAILLM)
            assert llm.model_name == self.model_id
            assert llm.base_url == "https://example.com"
            assert llm.api_key.get_secret_value() == "another.api.key"  # type: ignore

    @pytest.mark.asyncio
    async def test_agenerate(
        self, async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
        llm._aclient = async_openai_mock

        mocked_completion = Mock(
            choices=[
                Mock(
                    message=Mock(content=" Aenean hendrerit aliquam velit. ..."),
                    logprobs=Mock(
                        content=[
                            Mock(top_logprobs=[Mock(token=" ", logprob=-1)]),
                            Mock(top_logprobs=[Mock(token="Aenean", logprob=-2)]),
                        ]
                    ),
                )
            ],
            usage=Mock(prompt_tokens=100, completion_tokens=100),
        )
        llm._aclient.chat.completions.create = AsyncMock(return_value=mocked_completion)

        result = await llm.agenerate(
            input=[
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )
        assert result == {
            "generations": [" Aenean hendrerit aliquam velit. ..."],
            "statistics": {"input_tokens": [100], "output_tokens": [100]},
            "logprobs": [
                [[{"token": " ", "logprob": -1}], [{"token": "Aenean", "logprob": -2}]]
            ],
        }

    @pytest.mark.asyncio
    async def test_agenerate_with_string_input(
        self, async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
        llm._aclient = async_openai_mock

        mocked_completion = Mock(
            choices=[
                Mock(
                    text=" Aenean hendrerit aliquam velit. ...",
                    logprobs=Mock(top_logprobs=[{" ": -1}, {"Aenean": -2}]),
                )
            ],
            usage=Mock(prompt_tokens=100, completion_tokens=100),
        )
        llm._aclient.completions.create = AsyncMock(return_value=mocked_completion)

        result = await llm.agenerate(input="string input")
        assert result == {
            "generations": [" Aenean hendrerit aliquam velit. ..."],
            "statistics": {"input_tokens": [100], "output_tokens": [100]},
            "logprobs": [
                [[{"token": " ", "logprob": -1}], [{"token": "Aenean", "logprob": -2}]]
            ],
        }

    @pytest.mark.asyncio
    async def test_agenerate_structured(
        self, async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(
            model=self.model_id,
            api_key="api.key",
            structured_output={
                "schema": DummyUserDetail,
                "mode": "tool_call",
                "max_retries": 1,
            },
        )  # type: ignore
        llm._aclient = async_openai_mock

        mocked_usage = MagicMock(
            usage=MagicMock(prompt_tokens=100, completion_tokens=100),
        )
        sample_user = DummyUserDetail(
            name="John Doe", age=30, _raw_response=mocked_usage
        )

        llm._aclient.chat.completions.create = AsyncMock(return_value=sample_user)

        generation = await llm.agenerate(
            input=[
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )
        assert generation == {
            "generations": [sample_user.model_dump_json()],
            "statistics": {"input_tokens": [100], "output_tokens": [100]},
        }

    @pytest.mark.skipif(
        sys.version_info < (3, 9), reason="`mistralai` requires Python 3.9 or higher"
    )
    @pytest.mark.parametrize(
        "num_generations, expected_result",
        [
            (
                1,
                [
                    {
                        "generations": [" Aenean hendrerit aliquam velit. ..."],
                        "statistics": {"input_tokens": [100], "output_tokens": [100]},
                        "logprobs": [
                            [
                                [{"token": " ", "logprob": -1}],
                                [{"token": "Aenean", "logprob": -2}],
                            ]
                        ],
                    }
                ],
            ),
            (
                2,
                [
                    {
                        "generations": [" Aenean hendrerit aliquam velit. ..."] * 2,
                        "statistics": {"input_tokens": [100], "output_tokens": [100]},
                        "logprobs": [
                            [
                                [{"token": " ", "logprob": -1}],
                                [{"token": "Aenean", "logprob": -2}],
                            ]
                        ]
                        * 2,
                    }
                ],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_generate(
        self,
        async_openai_mock: MagicMock,
        _openai_mock: MagicMock,
        num_generations: int,
        expected_result: List[Dict[str, Any]],
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
        llm._aclient = async_openai_mock

        mocked_completion = Mock(
            choices=[
                Mock(
                    message=Mock(content=" Aenean hendrerit aliquam velit. ..."),
                    logprobs=Mock(
                        content=[
                            Mock(top_logprobs=[Mock(token=" ", logprob=-1)]),
                            Mock(top_logprobs=[Mock(token="Aenean", logprob=-2)]),
                        ]
                    ),
                )
            ]
            * num_generations,
            usage=Mock(prompt_tokens=100, completion_tokens=100),
        )
        llm._aclient.chat.completions.create = AsyncMock(return_value=mocked_completion)

        nest_asyncio.apply()

        result = llm.generate(
            inputs=[
                [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    },
                ]
            ]
        )
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_generate_with_string_input(
        self, async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
        llm._aclient = async_openai_mock

        mocked_completion = Mock(
            choices=[
                Mock(
                    text=" Aenean hendrerit aliquam velit. ...",
                    logprobs=Mock(top_logprobs=[{" ": -1}, {"Aenean": -2}]),
                )
            ],
            usage=Mock(prompt_tokens=100, completion_tokens=100),
        )
        llm._aclient.completions.create = AsyncMock(return_value=mocked_completion)

        nest_asyncio.apply()

        result = llm.generate(inputs=["input string"])
        assert result == [
            {
                "generations": [" Aenean hendrerit aliquam velit. ..."],
                "statistics": {"input_tokens": [100], "output_tokens": [100]},
                "logprobs": [
                    [
                        [{"token": " ", "logprob": -1}],
                        [{"token": "Aenean", "logprob": -2}],
                    ]
                ],
            }
        ]

    @pytest.mark.asyncio
    async def test_generate_raises_value_error_if_unknown_response_format(
        self, async_openai_mock: MagicMock, _: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
        llm._aclient = async_openai_mock

        with pytest.raises(ValueError):
            llm.generate(
                inputs=[
                    [
                        {"role": "system", "content": ""},
                        {
                            "role": "user",
                            "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                        },
                    ]
                ],
                response_format="unkown_format",
            )

    def test_offline_batch_generate(
        self, _async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
        llm._create_jobs = mock.MagicMock(return_value=("1234", "5678"))

        with pytest.raises(
            DistilabelOfflineBatchGenerationNotFinishedException
        ) as exception_info:
            llm.offline_batch_generate(
                inputs=[{"role": "user", "content": "How much is 2+2?"}]  # type: ignore
            )

        assert exception_info.value.jobs_ids == ("1234", "5678")

    def test_offline_batch_generate_with_job_ids(
        self, _async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key", jobs_ids=("1234",))  # type: ignore
        llm._check_and_get_batch_results = mock.MagicMock(
            return_value=[
                ["output 1"],
                ["output 2"],
            ]
        )
        assert llm.offline_batch_generate() == [["output 1"], ["output 2"]]

    def test_check_and_get_batch_results(
        self, async_openai_mock: MagicMock, openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key", jobs_ids=("1234",))  # type: ignore
        llm._aclient = async_openai_mock
        llm._client = openai_mock
        llm._retrieve_batch_results = mock.MagicMock(
            return_value=[
                {
                    "custom_id": 2,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "id": "1234",
                            "created": 13,
                            "model": "gpt-4",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "output 2",
                                    },
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 100,
                                "total_tokens": 200,
                            },
                        },
                    },
                },
                {
                    "custom_id": 1,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "id": "1234",
                            "created": 13,
                            "model": "gpt-4",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "output 1",
                                    },
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 100,
                                "total_tokens": 200,
                            },
                        },
                    },
                },
            ]
        )
        llm.load()

        outputs = llm._check_and_get_batch_results()

        assert outputs == [
            {
                "generations": ["output 1"],
                "statistics": {
                    "input_tokens": [100],
                    "output_tokens": [100],
                },
            },
            {
                "generations": ["output 2"],
                "statistics": {
                    "input_tokens": [100],
                    "output_tokens": [100],
                },
            },
        ]

    def test_check_and_get_batch_results_raises_valueerror(
        self, _async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore

        with pytest.raises(ValueError, match="No job IDs were found"):
            llm._check_and_get_batch_results()

    @pytest.mark.parametrize("status", ("validating", "in_progress", "finalizing"))
    def test_check_and_get_batch_results_raises_distilabel_exception(
        self, async_openai_mock: MagicMock, openai_mock: MagicMock, status: str
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key", jobs_ids=("1234",))  # type: ignore
        llm._aclient = async_openai_mock
        llm._client = openai_mock
        llm._get_openai_batch = mock.MagicMock(
            return_value=Batch(
                id="1234",
                completion_window="24h",
                created_at=13,
                endpoint="/v1/chat/completions",
                input_file_id="1234",
                object="batch",
                status=status,  # type: ignore
                output_file_id="1234",
            )
        )
        llm.load()

        with pytest.raises(DistilabelOfflineBatchGenerationNotFinishedException):
            llm._check_and_get_batch_results()

    @pytest.mark.parametrize("status", ("failed", "expired", "cancelled", "cancelling"))
    def test_check_and_get_batch_results_raises_runtimeerror(
        self, async_openai_mock: MagicMock, openai_mock: MagicMock, status: str
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key", jobs_ids=("1234",))  # type: ignore
        llm._aclient = async_openai_mock
        llm._client = openai_mock
        llm._get_openai_batch = mock.MagicMock(
            return_value=Batch(
                id="1234",
                completion_window="24h",
                created_at=13,
                endpoint="/v1/chat/completions",
                input_file_id="1234",
                object="batch",
                status=status,  # type: ignore
                output_file_id="1234",
            )
        )
        llm.load()

        with pytest.raises(
            RuntimeError,
            match=f"The only OpenAI API Batch that was created with ID '1234' failed with status '{status}",
        ):
            llm._check_and_get_batch_results()

    def test_parse_output(
        self, _async_openai_mock: MagicMock, openai_mock: MagicMock
    ) -> None:
        pass
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore

        result = llm._parse_output(
            {
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "1234",
                        "created": 13,
                        "model": "gpt-4",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": " Aenean hendrerit aliquam velit. ...",
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 100,
                            "total_tokens": 200,
                        },
                    },
                }
            }
        )

        assert result == {
            "generations": [" Aenean hendrerit aliquam velit. ..."],
            "statistics": {
                "input_tokens": [100],
                "output_tokens": [100],
            },
        }

    def test_retrieve_batch_results(
        self, _async_openai_mock: MagicMock, openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
        llm._client = openai_mock

        class Response:
            text: str = dedent(
                """
                {"response": {"status_code": 200, "body": {}}}
                {"response": {"status_code": 200, "body": {}}}
                {"response": {"status_code": 200, "body": {}}}
            """.lstrip()
            )

        llm._client.files.content.return_value = Response()

        results = llm._retrieve_batch_results(
            batch=Batch(
                id="1234",
                completion_window="24h",
                created_at=13,
                endpoint="/v1/chat/completions",
                input_file_id="1234",
                object="batch",
                status="completed",
                output_file_id="1234",
            )
        )  # type: ignore
        assert len(results) == 3

    def test_create_jobs(
        self, _async_openai_mock: MagicMock, openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
        llm._client = openai_mock

        messages = [
            {
                "role": "user",
                "content": "x" * ((_OPENAI_BATCH_API_MAX_FILE_SIZE // 100) - 50),
            }
        ]
        inputs = [messages] * 150

        jobs = llm._create_jobs(inputs=inputs)  # type: ignore
        assert isinstance(jobs, tuple)
        assert len(jobs) == 2

    def test_create_batch_files(
        self, _async_openai_mock: MagicMock, openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
        llm._client = openai_mock

        messages = [
            {
                "role": "user",
                "content": "x" * ((_OPENAI_BATCH_API_MAX_FILE_SIZE // 100) - 50),
            }
        ]
        inputs = [messages] * 150

        files = llm._create_batch_files(inputs=inputs)  # type: ignore
        assert len(files) == 2

    def test_create_jsonl_buffers(
        self, _async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore

        # This should be around 1MB
        messages = [
            {
                "role": "user",
                "content": "x" * ((_OPENAI_BATCH_API_MAX_FILE_SIZE // 100) - 50),
            }
        ]

        # Create an input that is larger than the max file size (150MB)
        inputs = [messages] * 150
        output = list(llm._create_jsonl_buffers(inputs=inputs))  # type: ignore
        assert len(output) == 2

    def test_create_jsonl_row(
        self, _async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
        output = llm._create_jsonl_row(
            input=[{"role": "user", "content": "How much is 2+2?"}],
            custom_id="unit-test",
            **{
                "model": "gpt-4",
                "temperature": 0.8,
                "max_new_tokens": 512,
            },
        )

        assert isinstance(output, bytes)
        assert orjson.loads(output.decode("utf-8")) == {
            "custom_id": "unit-test",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [
                    {
                        "role": "user",
                        "content": "How much is 2+2?",
                    }
                ],
                "model": "gpt-4",
                "temperature": 0.8,
                "max_new_tokens": 512,
            },
        }

    @pytest.mark.parametrize(
        "default_headers, structured_output, dump",
        [
            (
                None,
                None,
                {
                    "model": "gpt-4",
                    "generation_kwargs": {},
                    "max_retries": 6,
                    "base_url": "https://api.openai.com/v1",
                    "timeout": 120,
                    "default_headers": None,
                    "structured_output": None,
                    "jobs_ids": None,
                    "offline_batch_generation_block_until_done": None,
                    "use_offline_batch_generation": False,
                    "type_info": {
                        "module": "distilabel.models.llms.openai",
                        "name": "OpenAILLM",
                    },
                },
            ),
            (
                {"X-Custom-Header": "test"},
                {
                    "schema": DummyUserDetail.model_json_schema(),
                    "mode": "tool_call",
                    "max_retries": 1,
                },
                {
                    "model": "gpt-4",
                    "generation_kwargs": {},
                    "max_retries": 6,
                    "base_url": "https://api.openai.com/v1",
                    "timeout": 120,
                    "default_headers": {"X-Custom-Header": "test"},
                    "structured_output": {
                        "schema": DummyUserDetail.model_json_schema(),
                        "mode": "tool_call",
                        "max_retries": 1,
                    },
                    "jobs_ids": None,
                    "offline_batch_generation_block_until_done": None,
                    "use_offline_batch_generation": False,
                    "type_info": {
                        "module": "distilabel.models.llms.openai",
                        "name": "OpenAILLM",
                    },
                },
            ),
        ],
    )
    def test_serialization(
        self,
        _async_openai_mock: MagicMock,
        _openai_mock: MagicMock,
        default_headers: Dict[str, Any],
        structured_output: Dict[str, Any],
        dump: Dict[str, Any],
    ) -> None:
        llm = OpenAILLM(
            model=self.model_id,
            default_headers=default_headers,
            structured_output=structured_output,
        )

        assert llm.dump() == dump
        assert isinstance(OpenAILLM.from_dict(dump), OpenAILLM)

    def test_openai_llm_default_headers(
        self, _async_openai_mock: MagicMock, _openai_mock: MagicMock
    ) -> None:
        custom_headers = {"X-Custom-Header": "test"}
        llm = OpenAILLM(
            model=self.model_id, api_key="api.key", default_headers=custom_headers
        )  # type: ignore

        assert isinstance(llm, OpenAILLM)
        assert llm.model_name == self.model_id
        assert llm.default_headers == custom_headers
