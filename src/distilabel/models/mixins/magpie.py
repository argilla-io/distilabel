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

from typing import TYPE_CHECKING, Dict, Literal, Union

from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from distilabel.typing import StandardInput

MagpieAvailablePreQueryTemplates = Literal["llama3", "qwen2"]
"""The available predefined pre-query templates."""

MAGPIE_PRE_QUERY_TEMPLATES: Dict[MagpieAvailablePreQueryTemplates, str] = {
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    "qwen2": "<|im_start|>user\n",
}


class MagpieChatTemplateMixin(BaseModel, validate_assignment=True):
    """A simple mixin that adds the required logic to apply the pre-query template that
    allows to an instruct fine-tuned LLM to generate user instructions as described in
    the paper 'Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing'.

    This mixin is meant to be used in combination with the [Magpie][distilabel.steps.tasks.magpie.base.Magpie]
    task.

    Attributes:
        use_magpie_template: a flag used to enable/disable applying the Magpie pre-query
            template. Defaults to `False`.
        magpie_pre_query_template: the pre-query template to be applied to the prompt or
            sent to the LLM to generate an instruction or a follow up user message. Valid
            values are "llama3", "qwen2" or another pre-query template provided. Defaults
            to `None`.

    References:
        - [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
    """

    use_magpie_template: bool = False
    magpie_pre_query_template: Union[MagpieAvailablePreQueryTemplates, str, None] = None

    @field_validator("magpie_pre_query_template")
    @classmethod
    def magpie_pre_query_template_validator(cls, value: str) -> str:
        """Resolves the pre-query template alias if it exists, otherwise, returns the
        value with no modification."""
        if value in MAGPIE_PRE_QUERY_TEMPLATES:
            return MAGPIE_PRE_QUERY_TEMPLATES[value]
        return value

    @model_validator(mode="after")
    def use_magpie_template_validation(self) -> Self:
        """Checks that there is a pre-query template set if Magpie is going to be used."""
        if self.use_magpie_template and self.magpie_pre_query_template is None:
            raise ValueError(
                f"Cannot set `use_magpie_template=True` if `magpie_pre_query_template` is"
                f" `None`. To use Magpie with `{self.__class__.__name__}` you need to set"
                f" the `magpie_pre_query_template` attribute."
            )
        return self

    def apply_magpie_pre_query_template(
        self, prompt: str, input: "StandardInput"
    ) -> str:
        """Applies the pre-query template to the prompt if Magpie is going to be used.

        Args:
            prompt: the prompt to which the pre-query template has to be applied.
            input: the list with the chat items that were used to generate the prompt.

        Returns:
            The prompt with the pre-query template applied if needed.
        """
        if not self.use_magpie_template or (input and input[-1]["role"] == "user"):
            return prompt
        return prompt + self.magpie_pre_query_template  # type: ignore
