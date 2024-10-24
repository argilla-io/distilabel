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

from distilabel.models.llms.mixins.magpie import MAGPIE_PRE_QUERY_TEMPLATES
from tests.unit.conftest import DummyMagpieLLM


class TestMagpieChatTemplateMixin:
    def test_magpie_pre_query_template_set(self) -> None:
        with pytest.raises(
            ValueError,
            match="Cannot set `use_magpie_template=True` if `magpie_pre_query_template` is `None`",
        ):
            DummyMagpieLLM(use_magpie_template=True)

    def test_magpie_pre_query_template_alias_resolved(self) -> None:
        llm = DummyMagpieLLM(magpie_pre_query_template="llama3")
        assert llm.magpie_pre_query_template == MAGPIE_PRE_QUERY_TEMPLATES["llama3"]

    def test_apply_magpie_pre_query_template(self) -> None:
        llm = DummyMagpieLLM(magpie_pre_query_template="<user>")

        assert (
            llm.apply_magpie_pre_query_template(
                prompt="<system>Hello hello</system>", input=[]
            )
            == "<system>Hello hello</system>"
        )

        llm = DummyMagpieLLM(
            use_magpie_template=True, magpie_pre_query_template="<user>"
        )

        assert (
            llm.apply_magpie_pre_query_template(
                prompt="<system>Hello hello</system>", input=[]
            )
            == "<system>Hello hello</system><user>"
        )

        assert (
            llm.apply_magpie_pre_query_template(
                prompt="<system>Hello hello</system><user>Hey</user>",
                input=[{"role": "user", "content": "Hey"}],
            )
            == "<system>Hello hello</system><user>Hey</user>"
        )
