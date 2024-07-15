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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import Field, PositiveInt

from distilabel.llms.base import LLM
from distilabel.llms.mixins.magpie import MagpieChatTemplateMixin
from distilabel.mixins.runtime_parameters import (
    RuntimeParameter,
    RuntimeParametersMixin,
)
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType, FormattedInput
    from distilabel.steps.typing import StepOutput

MAGPIE_MULTI_TURN_SYSTEM_PROMPT = (
    "You are a helpful Al assistant. The user will engage in a multi−round conversation"
    " with you, asking initial questions and following up with additional related questions."
    " Your goal is to provide thorough, relevant and insightful responses to help the user"
    " with their queries."
)


class MagpieBase(RuntimeParametersMixin):
    """Base class defining the generation logic of Magpie method.

    References:
        - [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
    """

    llm: LLM

    n_turns: RuntimeParameter[PositiveInt] = Field(
        default=1,
        description="The number of turns to generate for the conversation.",
    )
    only_instruction: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to generate only the instruction. If this argument"
        " is `True`, then `n_turns` will be ignored.",
    )
    system_prompt: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="An optional system prompt that can be used to steer the LLM to generate"
        " content of certain topic, guide the style, etc.",
    )

    def _prepare_inputs_for_instruction_generation(
        self, inputs: List[Dict[str, Any]]
    ) -> List["FormattedInput"]:
        """Prepares the inputs adding the system (if required) prompt provided in each row,
        or if the conversations to generate have more than one turn, then adding the system
        prompt for multi-turn conversation from the paper.

        Args:
            inputs: the inputs to prepare.

        Returns:
            The prepared inputs.
        """
        prepared_inputs = []
        for input in inputs:
            conversation = []
            if "system_prompt" in input:
                conversation.append(
                    {"role": "system", "content": input["system_prompt"]}
                )
            elif self.system_prompt is not None:
                conversation.append({"role": "system", "content": self.system_prompt})
            elif self.n_turns > 1:  # type: ignore
                conversation.append(
                    {"role": "system", "content": MAGPIE_MULTI_TURN_SYSTEM_PROMPT}
                )

            prepared_inputs.append(conversation)

        return prepared_inputs

    def _append_messages_to_conversations(
        self, role: str, messages: List[str], conversations: List["ChatType"]
    ) -> List["ChatType"]:
        """Appends the outputs generated by the LLM with the specified role to the conversations.

        Args:
            role: the role to assign to the message to be appended.
            messages: the list of messages generated by the LLM for each conversation.
            conversations: the list of conversations to which the messages will be appended.

        Returns:
            The updated conversations.
        """
        for instruction, conversation in zip(messages, conversations):
            conversation.append({"role": role, "content": instruction})
        return conversations

    def _generate_multi_turn_conversation(
        self, inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        conversations = self._prepare_inputs_for_instruction_generation(inputs)

        for _ in range(self.n_turns):  # type: ignore
            # Generate instruction or user message
            outputs = self.llm.generate(
                inputs=conversations,
                num_generations=1,
                **self.llm.generation_kwargs,  # type: ignore
            )

            conversations = self._append_messages_to_conversations(
                role="user",
                messages=[output[0] for output in outputs],
                conversations=conversations,  # type: ignore
            )

            # TODO: handle potential previous `None`s

            # Generate response
            outputs = self.llm.generate(
                inputs=conversations,
                num_generations=1,
                **self.llm.generation_kwargs,  # type: ignore
            )

            conversations = self._append_messages_to_conversations(
                role="assistant",
                messages=[output[0] for output in outputs],
                conversations=conversations,  # type: ignore
            )

        return [
            {**input, "conversation": conversation}
            for input, conversation in zip(inputs, conversations)
        ]

    def _generate_with_pre_query_template(
        self, inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate a list of instructions or conversations of the specified number of turns.

        Args:
            inputs: a list of dictionaries that can contain a `system_prompt` key.

        Returns:
            The list of generated conversations.
        """

        if self.only_instruction:
            prepared_inputs = self._prepare_inputs_for_instruction_generation(inputs)
            outputs = self.llm.generate(
                inputs=prepared_inputs,
                num_generations=1,
                **self.llm.generation_kwargs,  # type: ignore
            )
            return [
                {**input, "instruction": output[0]}
                for input, output in zip(inputs, outputs)
            ]

        return self._generate_multi_turn_conversation(inputs)


class Magpie(Task, MagpieBase):
    """Generates conversations using an instruct fine-tuned LLM.

    Magpie is a neat method that allows generating user instructions with no seed data
    or specific system prompt thanks to the autoregressive capabilities of the instruct
    fine-tuned LLMs. As they were fine-tuned using a chat template composed by a user message
    and a desired assistant output, the instruct fine-tuned LLM learns that after the pre-query
    or pre-instruct tokens comes an instruction. If these pre-query tokens are sent to the
    LLM without any user message, then the LLM will continue generating tokens as if it was
    the user. This trick allows "extracting" instructions from the instruct fine-tuned LLM.
    After this instruct is generated, it can be sent again to the LLM to generate this time
    an assistant response. This process can be repeated N times allowing to build a multi-turn
    conversation. This method was described in the paper 'Magpie: Alignment Data Synthesis from
    Scratch by Prompting Aligned LLMs with Nothing'.

    Attributes:
        n_turns: the number of turns that the generated conversation will have.
        only_instruction: whether to generate only the instruction. If this argument is
            `True`, then `n_turns` will be ignored. Defaults to `False`.
        system_prompt: an optional system prompt that can be used to steer the LLM to generate
            content of certain topic, guide the style, etc. If the provided inputs contains
            a `system_prompt` column, then this runtime parameter will be ignored and the
            one from the column will be used. Defaults to `None`.

    Runtime parameters:
        - `n_turns`: the number of turns that the generated conversation will have.
        - `only_instruction`: whether to generate only the instruction. If this argument is
            `True`, then `n_turns` will be ignored. Defaults to `False`.
        - `system_prompt`: an optional system prompt that can be used to steer the LLM to
            generate content of certain topic, guide the style, etc. If the provided inputs
            contains a `system_prompt` column, then this runtime parameter will be ignored
            and the one from the column will be used. Defaults to `None`.

    Input columns:
        - system_prompt (`str`, optional): an optional system prompt that can be provided
            to guide the generation of the instruct LLM and steer it to generate instructions
            of certain topic.

    Output columns:
        - conversation (`ChatType`): the generated conversation which is a list of chat
            items with a role and a message. Only if `only_instructions=False`.
        - instruction (`str`): the generated instructions if `only_instruction=True`.

    Categories:
        - text-generation
        - instruction

    References:
        - [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)

    Examples:

        Generating instructions with Llama 3 8B Instruct and TransformersLLM:

        ```python
        from distilabel.llms import TransformersLLM
        from distilabel.steps.tasks import Magpie

        magpie = Magpie(
            llm=TransformersLLM(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 1.0,
                    "max_new_tokens": 64,
                },
                device="mps",
            ),
            only_instruction=True,
        )

        magpie.load()

        result = next(
            magpie.process(
                inputs=[
                    {
                        "system_prompt": "You're a math expert AI assistant that helps students of secondary school to solve calculus problems."
                    },
                    {
                        "system_prompt": "You're an expert florist AI assistant that helps user to erradicate pests in their crops."
                    },
                ]
            )
        )
        # [
        #     {'instruction': "That's me! I'd love some help with solving calculus problems! What kind of calculation are you most effective at? Linear Algebra, derivatives, integrals, optimization?"},
        #     {'instruction': 'I was wondering if there are certain flowers and plants that can be used for pest control?'}
        # ]
        ```

        Generating conversations with Llama 3 8B Instruct and TransformersLLM:

        ```python
        from distilabel.llms import TransformersLLM
        from distilabel.steps.tasks import Magpie

        magpie = Magpie(
            llm=TransformersLLM(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 1.0,
                    "max_new_tokens": 256,
                },
                device="mps",
            ),
            n_turns=2,
        )

        magpie.load()

        result = next(
            magpie.process(
                inputs=[
                    {
                        "system_prompt": "You're a math expert AI assistant that helps students of secondary school to solve calculus problems."
                    },
                    {
                        "system_prompt": "You're an expert florist AI assistant that helps user to erradicate pests in their crops."
                    },
                ]
            )
        )
        # [
        #     {
        #         'conversation': [
        #             {'role': 'system', 'content': "You're a math expert AI assistant that helps students of secondary school to solve calculus problems."},
        #             {
        #                 'role': 'user',
        #                 'content': 'I\'m having trouble solving the limits of functions in calculus. Could you explain how to work with them? Limits of functions are denoted by lim x→a f(x) or lim x→a [f(x)]. It is read as "the limit as x approaches a of f
        # of x".'
        #             },
        #             {
        #                 'role': 'assistant',
        #                 'content': 'Limits are indeed a fundamental concept in calculus, and understanding them can be a bit tricky at first, but don\'t worry, I\'m here to help! The notation lim x→a f(x) indeed means "the limit as x approaches a of f of
        # x". What it\'s asking us to do is find the'
        #             }
        #         ]
        #     },
        #     {
        #         'conversation': [
        #             {'role': 'system', 'content': "You're an expert florist AI assistant that helps user to erradicate pests in their crops."},
        #             {
        #                 'role': 'user',
        #                 'content': "As a flower shop owner, I'm noticing some unusual worm-like creatures causing damage to my roses and other flowers. Can you help me identify what the problem is? Based on your expertise as a florist AI assistant, I think it
        # might be pests or diseases, but I'm not sure which."
        #             },
        #             {
        #                 'role': 'assistant',
        #                 'content': "I'd be delighted to help you investigate the issue! Since you've noticed worm-like creatures damaging your roses and other flowers, I'll take a closer look at the possibilities. Here are a few potential culprits: 1.
        # **Aphids**: These small, soft-bodied insects can secrete a sticky substance called"
        #             }
        #         ]
        #     }
        # ]
        ```
    """

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
    def inputs(self) -> List[str]:
        return []

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """Does nothing."""
        return []

    @property
    def outputs(self) -> List[str]:
        """Either a multi-turn conversation or the instruction generated."""
        if self.only_instruction:
            return ["instruction"]
        return ["conversation"]

    def format_output(
        self,
        output: Union[str, None],
        input: Union[Dict[str, Any], None] = None,
    ) -> Dict[str, Any]:
        """Does nothing."""
        return {}

    def process(self, inputs: StepInput) -> "StepOutput":
        """Generate a list of instructions or conversations of the specified number of turns.

        Args:
            inputs: a list of dictionaries that can contain a `system_prompt` key.

        Yields:
            The list of generated conversations.
        """
        yield self._generate_with_pre_query_template(inputs)