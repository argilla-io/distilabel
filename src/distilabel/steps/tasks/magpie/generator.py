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

from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import Field

from distilabel.llms.mixins.magpie import MagpieChatTemplateMixin
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.base import GeneratorTask
from distilabel.steps.tasks.magpie.base import MagpieBase

if TYPE_CHECKING:
    from distilabel.steps.typing import GeneratorStepOutput


class MagpieGenerator(GeneratorTask, MagpieBase):
    """Generator task the generates instructions or conversations using Magpie.

    Magpie is a neat method that allows generating user instructions with no seed data
    or specific system prompt thanks to the autoregressive capabilities of the instruct
    fine-tuned LLMs. As they were fine-tuned using a chat template composed by a user message
    and a desired assistant output, the instruct fine-tuned LLM learns that after the pre-query
    or pre-instruct tokens comes an instruction. If these pre-query tokens are sent to the
    LLM without any user message, then the LLM will continue generating tokens as it was
    the user. This trick allows "extracting" instructions from the instruct fine-tuned LLM.
    After this instruct is generated, it can be sent again to the LLM to generate this time
    an assistant response. This process can be repeated N times allowing to build a multi-turn
    conversation. This method was described in the paper 'Magpie: Alignment Data Synthesis from
    Scratch by Prompting Aligned LLMs with Nothing'.

    Runtime parameters:
        - `n_turns`: the number of turns that the generated conversation will have.
        - `system_prompt`: an optional system prompt that can be used to steer the LLM to
            generate content of certain topic, guide the style, etc. Defaults to `None`.
        - `num_rows`: the number of rows to be generated.

    Output columns:
        - conversation (`ChatType`): the generated conversation which is a list of chat
            items with a role and a message.

    Categories:
        - instruction-generation
        - mt-generation
        - generator

    References:
        - [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
    """

    # TODO: move this to `GeneratorTask`
    num_rows: RuntimeParameter[int] = Field(
        default=None, description="The number of rows to generate."
    )

    def model_post_init(self, __context: Any) -> None:
        """Checks that the provided `LLM` uses the `MagpieChatTemplateMixin`."""
        super().model_post_init(__context)

        if not isinstance(self.llm, MagpieChatTemplateMixin):
            raise ValueError(
                f"`Magpie` task can only be used with an `LLM` that uses the `MagpieChatTemplateMixin`."
                f"`{self.llm.__class__.__name__}` doesn't use the aforementioned mixin."
            )

        self.llm.use_magpie_template = True

    @property
    def outputs(self) -> List[str]:
        return ["conversation"]

    def format_output(
        self,
        output: Union[str, None],
        input: Union[Dict[str, Any], None] = None,
    ) -> Dict[str, Any]:
        """Does nothing."""
        return {}

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        """Generates the desired number of instructions or conversations using Magpie.

        Args:
            offset: The offset to start the generation from. Defaults to `0`.

        Yields:
            The generated instructions or conversations.
        """
        generated = offset

        while generated <= self.num_rows:  # type: ignore
            rows_to_generate = (
                self.num_rows if self.num_rows < self.batch_size else self.batch_size  # type: ignore
            )
            conversations = self._generate_with_pre_query_template(
                inputs=[{} for _ in range(rows_to_generate)]  # type: ignore
            )
            generated += rows_to_generate  # type: ignore
            yield (conversations, generated == self.num_rows)
