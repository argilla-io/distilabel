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

import inspect

import pytest

pytest.importorskip("vllm")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.sampling_params import StructuredOutputsParams  # noqa: E402

from distilabel.models.llms import vLLM  # noqa: E402


def test_vllm_api_compatibility() -> None:
    """Check the real vLLM API surface used by the distilabel integration."""
    llm_parameters = inspect.signature(LLM).parameters
    for parameter in (
        "model",
        "dtype",
        "trust_remote_code",
        "quantization",
        "revision",
        "tokenizer",
        "tokenizer_mode",
        "tokenizer_revision",
        "skip_tokenizer_init",
        "seed",
    ):
        assert parameter in llm_parameters

    generate_parameters = inspect.signature(LLM.generate).parameters
    assert {"prompts", "sampling_params", "use_tqdm"}.issubset(generate_parameters)
    assert callable(destroy_model_parallel)
    assert callable(destroy_distributed_environment)

    structured_output = StructuredOutputsParams(regex=r"[0-9]+")
    sampling_params = SamplingParams(
        n=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=4,
        prompt_logprobs=None,
        logprobs=None,
        stop=None,
        stop_token_ids=None,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        structured_outputs=structured_output,
    )
    assert sampling_params.structured_outputs == structured_output

    llm = vLLM(model="unused", disable_cuda_device_placement=True)
    json_output = llm._prepare_structured_output_params(
        {
            "format": "json",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        }
    )
    regex_output = llm._prepare_structured_output_params(
        {"format": "regex", "schema": r"[0-9]+"}
    )
    assert isinstance(json_output, StructuredOutputsParams)
    assert isinstance(regex_output, StructuredOutputsParams)


@pytest.mark.timeout(180)
def test_vllm_cpu_generation() -> None:
    """Load a tiny model and exercise distilabel's vLLM generation and cleanup paths."""
    llm = vLLM(
        model="distilabel-internal-testing/tiny-random-mistral",
        dtype="float32",
        disable_cuda_device_placement=True,
        extra_kwargs={
            "enforce_eager": True,
            "gpu_memory_utilization": 0.1,
            "max_model_len": 64,
            "max_num_batched_tokens": 64,
            "max_num_seqs": 1,
        },
    )

    try:
        llm.load()
        outputs = llm.generate(
            inputs=["Hello"],
            max_new_tokens=4,
            temperature=0.0,
            extra_sampling_params={"ignore_eos": True},
        )
    finally:
        llm.unload()

    assert len(outputs) == 1
    assert len(outputs[0]["generations"]) == 1
    assert outputs[0]["statistics"]["output_tokens"] == [4]
