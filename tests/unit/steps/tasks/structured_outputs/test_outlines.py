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
import os
from typing import Any, List, Union

import pytest
from distilabel.llms.huggingface.transformers import TransformersLLM
from distilabel.llms.openai import OpenAILLM
from distilabel.steps.tasks.structured_outputs.outlines import OutlinesStructuredOutput
from outlines.models.openai import OpenAI
from pydantic import BaseModel

DISTILABEL_RUN_SLOW_TESTS = os.getenv("DISTILABEL_RUN_SLOW_TESTS", False)


@pytest.fixture(scope="module")
def tiny_mistral_llm() -> TransformersLLM:
    llm = TransformersLLM(model="openaccess-ai-collective/tiny-mistral")
    # llm = TransformersLLM(model="Locutusque/TinyMistral-248M-v2.5-Instruct")
    llm.load()
    return llm


@pytest.fixture(scope="module")
def tiny_mistral_llm_structured() -> TransformersLLM:
    llm = TransformersLLM(
        model="openaccess-ai-collective/tiny-mistral",
        structured_output={"format": "text", "structure": None},
    )
    llm.load()
    return llm


class DummyUserTest(BaseModel):
    name: str
    last_name: str
    id: int


def model_to_schema(model: BaseModel) -> str:
    schema = model.model_json_schema()
    schema.pop("required")
    return json.dumps(schema)


class TestOutlinesStructuredOutput:
    # def test_wrong_output_format(self):
    #     with pytest.raises(NotImplementedError):
    #         OutlinesStructuredOutput(llm=DummyLLM())

    # def test_not_loaded_llm(self):
    #     with pytest.raises(ValueError):
    #         OutlinesStructuredOutput(llm="not_allowed")

    # Will only test the OpenAI and Transformers for now, as vLLM is not supported in macos, and llamacpp
    # is implemented really similar to Transformers in outlines.

    @pytest.mark.parametrize(
        "output_format, expected",
        [
            ("text", OpenAI),
            ("json", None),
            ("regex", None),
            ("cfg", None),
        ],
    )
    def test_openai_instance(
        self, output_format: str, expected: Union[OpenAI, None]
    ) -> None:
        model = "gpt-3.5-turbo"
        llm = OpenAILLM(model=model)
        if output_format == "text":
            structured_output = OutlinesStructuredOutput.from_openai(
                llm=llm, output_format=output_format
            )
            structured_output.load()
            assert isinstance(structured_output._structured_generator, expected)
        else:
            with pytest.raises(NotImplementedError):
                OutlinesStructuredOutput.from_openai(
                    llm=llm, output_format=output_format
                )

    @pytest.mark.skipif(
        not DISTILABEL_RUN_SLOW_TESTS,
        reason="Slow tests, run locally when needed.",
    )
    @pytest.mark.parametrize(
        "input",
        [
            "What is 2+2?",
            ["What is 2+2?"],
        ],
    )
    def test_transformers_text_single_and_batch(
        self, tiny_mistral_llm: TransformersLLM, input: Union[str, List[str]]
    ) -> None:
        structured_output = OutlinesStructuredOutput.from_transformers(
            llm=tiny_mistral_llm, output_format="text"
        )
        structured_output.load()
        result = structured_output(input, max_tokens=10)
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)

    @pytest.mark.parametrize(
        "output_format",
        [
            "text",
            "json",
            "regex",
            "cfg",
        ],
    )
    @pytest.mark.parametrize(
        "sampler",
        ["multinomial", "greedy", "beam"],
    )
    def test_samplers(
        self, tiny_mistral_llm: TransformersLLM, output_format: str, sampler: str
    ) -> None:
        if output_format == "text":
            structured_output = OutlinesStructuredOutput.from_transformers(
                llm=tiny_mistral_llm, output_format=output_format, sampler=sampler
            )
            structured_output.load()
            structured_output._structured_generator.sampler = sampler
        elif output_format == "regex":
            structured_output = OutlinesStructuredOutput.from_transformers(
                llm=tiny_mistral_llm,
                output_format=output_format,
                # Some random regex pattern
                output_structure=r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
                sampler=sampler,
            )
            structured_output.load()
            structured_output._structured_generator.sampler = sampler

        else:
            with pytest.raises(NotImplementedError):
                structured_output = OutlinesStructuredOutput.from_transformers(
                    llm=tiny_mistral_llm, output_format=output_format, sampler=sampler
                )
                structured_output.load()

    @pytest.mark.skipif(
        not DISTILABEL_RUN_SLOW_TESTS,
        reason="Slow tests, run locally when needed.",
    )
    def test_transformers_regex_single(self, tiny_mistral_llm: TransformersLLM) -> None:
        pattern = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
        structured_output = OutlinesStructuredOutput.from_transformers(
            llm=tiny_mistral_llm, output_format="regex", output_structure=pattern
        )
        structured_output.load()

        result = structured_output(
            "What is the IP address of the Google DNS servers? ", max_tokens=30
        )
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)

    @pytest.mark.skipif(
        not DISTILABEL_RUN_SLOW_TESTS,
        reason="Slow tests, run locally when needed.",
    )
    @pytest.mark.parametrize(
        "schema",
        [
            DummyUserTest,
            model_to_schema(DummyUserTest),
        ],
    )
    def test_transformers_json_single(
        self, schema: Any, tiny_mistral_llm: TransformersLLM
    ) -> None:
        structured_output = OutlinesStructuredOutput.from_transformers(
            llm=tiny_mistral_llm, output_format="json", output_structure=schema
        )
        structured_output.load()
        result = structured_output(
            "Create a user profile with the fields name, last_name and id",
            max_tokens=50,
        )
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)

    def test_transformers_json_mode(self, tiny_mistral_llm: TransformersLLM) -> None:
        # Asking for json without an output structure is equivalent to working with json mode,
        # which in outlines means working with "cfg" and a special grammar.
        # NOTE: This should work, there's a bug on outlines side
        with pytest.raises(NotImplementedError):
            structured_output = OutlinesStructuredOutput.from_transformers(
                llm=tiny_mistral_llm,
                output_format="json",
            )
            structured_output.load()
            result = structured_output("Generate some JSON", max_tokens=50)
            assert isinstance(result, list)
            assert isinstance(result[0], list)
            assert isinstance(result[0][0], str)

    def test_serialization(self, tiny_mistral_llm: TransformersLLM) -> None:
        structured_output = OutlinesStructuredOutput.from_transformers(
            llm=tiny_mistral_llm, output_format="text"
        )
        structured_output.load()
        assert structured_output.dump() == {
            "sampler": "multinomial",
            "output_format": "text",
            "output_structure": None,
            "whitespace_pattern": None,
            "num_generations": 1,
            "top_k": None,
            "top_p": None,
            "temperature": None,
            "type_info": {
                "module": "distilabel.steps.tasks.structured_outputs.outlines",
                "name": "OutlinesStructuredOutput",
            },
        }

        # with pytest.raises(NotImplementedError):
        #     structured_output = OutlinesStructuredOutput.from_dict({})
        #     structured_output.load()


class TestOutlinesFromLLM:
    DUMP = {
        "cuda_devices": "auto",
        "generation_kwargs": {},
        "structured_output": {
            "sampler": "multinomial",
            "output_format": "text",
            "output_structure": None,
            "whitespace_pattern": None,
            "num_generations": 1,
            "top_k": None,
            "top_p": None,
            "temperature": None,
            "type_info": {
                "module": "distilabel.steps.tasks.structured_outputs.outlines",
                "name": "OutlinesStructuredOutput",
            },
        },
        "model": "openaccess-ai-collective/tiny-mistral",
        "revision": "main",
        "torch_dtype": "auto",
        "trust_remote_code": False,
        "model_kwargs": None,
        "tokenizer": None,
        "use_fast": True,
        "chat_template": None,
        "device": None,
        "device_map": None,
        "token": None,
        "type_info": {
            "module": "distilabel.llms.huggingface.transformers",
            "name": "TransformersLLM",
        },
    }

    # Test that we can instantiate it with multiple formats
    # Test generation

    @pytest.mark.skipif(
        not DISTILABEL_RUN_SLOW_TESTS,
        reason="Slow tests, run locally when needed.",
    )
    def test_serialization(self, tiny_mistral_llm_structured: TransformersLLM) -> None:
        assert tiny_mistral_llm_structured.dump() == self.DUMP

    @pytest.mark.skipif(
        not DISTILABEL_RUN_SLOW_TESTS,
        reason="Slow tests, run locally when needed.",
    )
    def test_load_from_dict(self) -> None:
        llm = TransformersLLM.from_dict(self.DUMP)
        assert isinstance(llm, TransformersLLM)
        llm.load()
        assert llm._structured_generator is not None

    @pytest.mark.parametrize(
        "format, structure, prompt",
        [
            ("text", None, "What is 2+2?"),
            # ("json", None, "prompt"),  # JSON Mode (not working due to errors on cfg)
            (
                "json",
                DummyUserTest,
                "Create a user profile with the fields name, last_name and id",
            ),  #
            (
                "json",
                model_to_schema(DummyUserTest),
                "Create a user profile with the fields name, last_name and id",
            ),
            (
                "regex",
                r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
                "What is the IP address of the Google DNS servers?",
            ),
            # ("cfg", None),  # Not working due to errors on cfg
        ],
    )
    def test_structured_generation_from_dict(
        self, format: str, structure: Union[str, BaseModel, None], prompt: str
    ) -> None:
        llm = TransformersLLM(
            model="openaccess-ai-collective/tiny-mistral",
            structured_output={"format": format, "structure": structure},
        )
        llm.load()
        prompt = [
            [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
        ]
        result = llm.generate(prompt, max_new_tokens=30)
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)
