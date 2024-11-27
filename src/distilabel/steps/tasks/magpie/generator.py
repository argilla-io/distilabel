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

from typing import TYPE_CHECKING, Any, Dict, Union

from pydantic import Field
from typing_extensions import override

from distilabel.errors import DistilabelUserError
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.mixins.magpie import MagpieChatTemplateMixin
from distilabel.steps.tasks.base import GeneratorTask
from distilabel.steps.tasks.magpie.base import MagpieBase

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
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

    Attributes:
        n_turns: the number of turns that the generated conversation will have.
            Defaults to `1`.
        end_with_user: whether the conversation should end with a user message.
            Defaults to `False`.
        include_system_prompt: whether to include the system prompt used in the generated
            conversation. Defaults to `False`.
        only_instruction: whether to generate only the instruction. If this argument is
            `True`, then `n_turns` will be ignored. Defaults to `False`.
        system_prompt: an optional system prompt, or a list of system prompts from which
            a random one will be chosen, or a dictionary of system prompts from which a
            random one will be choosen, or a dictionary of system prompts with their probability
            of being chosen. The random system prompt will be chosen per input/output batch.
            This system prompt can be used to guide the generation of the instruct LLM and
            steer it to generate instructions of a certain topic. Defaults to `None`.
        num_rows: the number of rows to be generated.

    Runtime parameters:
        - `n_turns`: the number of turns that the generated conversation will have. Defaults
            to `1`.
        - `end_with_user`: whether the conversation should end with a user message.
            Defaults to `False`.
        - `include_system_prompt`: whether to include the system prompt used in the generated
            conversation. Defaults to `False`.
        - `only_instruction`: whether to generate only the instruction. If this argument is
            `True`, then `n_turns` will be ignored. Defaults to `False`.
        - `system_prompt`: an optional system prompt, or a list of system prompts from which
            a random one will be chosen, or a dictionary of system prompts from which a
            random one will be choosen, or a dictionary of system prompts with their probability
            of being chosen. The random system prompt will be chosen per input/output batch.
            This system prompt can be used to guide the generation of the instruct LLM and
            steer it to generate instructions of a certain topic.
        - `num_rows`: the number of rows to be generated.

    Output columns:
        - conversation (`ChatType`): the generated conversation which is a list of chat
            items with a role and a message.
        - instruction (`str`): the generated instructions if `only_instruction=True`.
        - response (`str`): the generated response if `n_turns==1`.
        - system_prompt_key (`str`, optional): the key of the system prompt used to generate
            the conversation or instruction. Only if `system_prompt` is a dictionary.
        - model_name (`str`): The model name used to generate the `conversation` or `instruction`.

    Categories:
        - text-generation
        - instruction
        - generator

    References:
        - [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)

    Examples:
        Generating instructions with Llama 3 8B Instruct and TransformersLLM:

        ```python
        from distilabel.models import TransformersLLM
        from distilabel.steps.tasks import MagpieGenerator

        generator = MagpieGenerator(
            llm=TransformersLLM(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 1.0,
                    "max_new_tokens": 256,
                },
                device="mps",
            ),
            only_instruction=True,
            num_rows=5,
        )

        generator.load()

        result = next(generator.process())
        # (
        #       [
        #           {"instruction": "I've just bought a new phone and I're excited to start using it."},
        #           {"instruction": "What are the most common types of companies that use digital signage?"}
        #       ],
        #       True
        # )
        ```

        Generating a conversation with Llama 3 8B Instruct and TransformersLLM:

        ```python
        from distilabel.models import TransformersLLM
        from distilabel.steps.tasks import MagpieGenerator

        generator = MagpieGenerator(
            llm=TransformersLLM(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 1.0,
                    "max_new_tokens": 64,
                },
                device="mps",
            ),
            n_turns=3,
            num_rows=5,
        )

        generator.load()

        result = next(generator.process())
        # (
        #     [
        #         {
        #             'conversation': [
        #                 {
        #                     'role': 'system',
        #                     'content': 'You are a helpful Al assistant. The user will engage in a multi−round conversation with you,asking initial questions and following up with additional related questions. Your goal is to provide thorough, relevant and
        # insightful responses to help the user with their queries.'
        #                 },
        #                 {'role': 'user', 'content': "I'm considering starting a social media campaign for my small business and I're not sure where to start. Can you help?"},
        #                 {
        #                     'role': 'assistant',
        #                     'content': "Exciting endeavor! Creating a social media campaign can be a great way to increase brand awareness, drive website traffic, and ultimately boost sales. I'd be happy to guide you through the process. To get started,
        # let's break down the basics. First, we need to identify your goals and target audience. What do"
        #                 },
        #                 {
        #                     'role': 'user',
        #                     'content': "Before I start a social media campaign, what kind of costs ammol should I expect to pay? There are several factors that contribute to the total cost of running a social media campaign. Let me outline some of the main
        # expenses you might encounter: 1. Time: As the business owner, you'll likely spend time creating"
        #                 },
        #                 {
        #                     'role': 'assistant',
        #                     'content': 'Time is indeed one of the biggest investments when it comes to running a social media campaign! Besides time, you may also incur costs associated with: 2. Content creation: You might need to hire freelancers or
        # agencies to create high-quality content (images, videos, captions) for your social media platforms. 3. Advertising'
        #                 }
        #             ]
        #         },
        #         {
        #             'conversation': [
        #                 {
        #                     'role': 'system',
        #                     'content': 'You are a helpful Al assistant. The user will engage in a multi−round conversation with you,asking initial questions and following up with additional related questions. Your goal is to provide thorough, relevant and
        # insightful responses to help the user with their queries.'
        #                 },
        #                 {'role': 'user', 'content': "I am thinking of buying a new laptop or computer. What are some important factors I should consider when making your decision? I'll make sure to let you know if any other favorites or needs come up!"},
        #                 {
        #                     'role': 'assistant',
        #                     'content': 'Exciting times ahead! When considering a new laptop or computer, there are several key factors to think about to ensure you find the right one for your needs. Here are some crucial ones to get you started: 1.
        # **Purpose**: How will you use your laptop or computer? For work, gaming, video editing,'
        #                 },
        #                 {
        #                     'role': 'user',
        #                     'content': 'Let me stop you there. Let\'s explore this "purpose" factor that you mentioned earlier. Can you elaborate more on what type of devices would be suitable for different purposes? For example, if I\'re primarily using my
        # laptop for general usage like browsing, email, and word processing, would a budget-friendly laptop be sufficient'
        #                 },
        #                 {
        #                     'role': 'assistant',
        #                     'content': "Understanding your purpose can greatly impact the type of device you'll need. **General Usage (Browsing, Email, Word Processing)**: For casual users who mainly use their laptop for daily tasks, a budget-friendly
        # option can be sufficient. Look for laptops with: * Intel Core i3 or i5 processor* "
        #                 }
        #             ]
        #         }
        #     ],
        #     True
        # )
        ```

        Generating with system prompts with probabilities:

        ```python
        from distilabel.models import InferenceEndpointsLLM
        from distilabel.steps.tasks import MagpieGenerator

        magpie = MagpieGenerator(
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3-8B-Instruct",
                tokenizer_id="meta-llama/Meta-Llama-3-8B-Instruct",
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 0.8,
                    "max_new_tokens": 256,
                },
            ),
            n_turns=2,
            system_prompt={
                "math": ("You're an expert AI assistant.", 0.8),
                "writing": ("You're an expert writing assistant.", 0.2),
            },
        )

        magpie.load()

        result = next(magpie.process())
        ```

    Citations:
        ```
        @misc{xu2024magpiealignmentdatasynthesis,
            title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing},
            author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Yuntian Deng and Radha Poovendran and Yejin Choi and Bill Yuchen Lin},
            year={2024},
            eprint={2406.08464},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2406.08464},
        }
        ```
    """

    # TODO: move this to `GeneratorTask`
    num_rows: RuntimeParameter[int] = Field(
        default=None, description="The number of rows to generate."
    )

    def model_post_init(self, __context: Any) -> None:
        """Checks that the provided `LLM` uses the `MagpieChatTemplateMixin`."""
        super().model_post_init(__context)

        if not isinstance(self.llm, MagpieChatTemplateMixin):
            raise DistilabelUserError(
                f"`Magpie` task can only be used with an `LLM` that uses the `MagpieChatTemplateMixin`."
                f"`{self.llm.__class__.__name__}` doesn't use the aforementioned mixin.",
                page="components-gallery/tasks/magpiegenerator/",
            )

        self.llm.use_magpie_template = True

        outputs = []

        if self.only_instruction:
            outputs.append("instruction")
        elif self.n_turns == 1:
            outputs.extend(["instruction", "response"])
        else:
            outputs.append("conversation")

        if isinstance(self.system_prompt, dict):
            outputs.append("system_prompt_key")

        outputs.append("model_name")

        self.outputs = outputs

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

    @override
    def _sample_input(self) -> "ChatType":
        return self._generate_with_pre_query_template(inputs=[{}])
