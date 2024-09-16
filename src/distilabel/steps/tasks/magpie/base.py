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

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from pydantic import Field, PositiveInt, field_validator

from distilabel.errors import DistilabelUserError
from distilabel.llms.base import LLM
from distilabel.llms.mixins.magpie import MagpieChatTemplateMixin
from distilabel.mixins.runtime_parameters import (
    RuntimeParameter,
    RuntimeParametersMixin,
)
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns, StepOutput

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

    llm: LLM

    n_turns: RuntimeParameter[PositiveInt] = Field(
        default=1,
        description="The number of turns to generate for the conversation.",
    )
    end_with_user: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether the conversation should end with a user message.",
    )
    include_system_prompt: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to include the system prompt used in the generated conversation.",
    )
    only_instruction: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to generate only the instruction. If this argument"
        " is `True`, then `n_turns` will be ignored.",
    )
    system_prompt: Optional[
        RuntimeParameter[
            Union[List[str], Dict[str, str], Dict[str, Tuple[str, float]], str]
        ]
    ] = Field(
        default=None,
        description="An optional system prompt, or a list of system prompts from which a"
        " random one will be chosen, or a dictionary of system prompts from which a random"
        " one will be choosen, or a dictionary of system prompts with their probability of"
        " being chosen. The random system prompt will be chosen per input/output batch."
        " This system prompt can be used to guide the generation of the instruct LLM and"
        " steer it to generate instructions of a certain topic.",
    )

    @field_validator("system_prompt", mode="after")
    @classmethod
    def system_prompts_weights_validator(
        cls,
        system_prompts: Union[
            List[str], Dict[str, str], Dict[str, Tuple[str, float]], str
        ],
    ) -> Union[List[str], Dict[str, str], Dict[str, Tuple[str, float]], str]:
        """Validates that the sum of the weights of the system prompts is equal to 1.0."""
        if isinstance(system_prompts, dict):
            system_prompts_values = list(system_prompts.values())
            if isinstance(system_prompts_values[0], tuple):
                weights_sum = sum(weight for _, weight in system_prompts_values)  # type: ignore
                if weights_sum != 1.0:
                    raise DistilabelUserError(
                        "If `system_prompts` attribute is a dictionary containing tuples with"
                        " the system prompts and their probability of being chosen, then the"
                        " sum of the weights must be equal to 1.0."
                    )
        return system_prompts

    @property
    def outputs(self) -> "StepColumns":
        """Either a multi-turn conversation or the instruction generated."""
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

        return outputs

    def _prepare_inputs_for_instruction_generation(
        self, inputs: List[Dict[str, Any]]
    ) -> Tuple[List["ChatType"], Union[str, None]]:
        """Prepares the inputs adding the system (if required) prompt provided in each row,
        or if the conversations to generate have more than one turn, then adding the system
        prompt for multi-turn conversation from the paper.

        Args:
            inputs: the inputs to prepare.

        Returns:
            The prepared inputs.
        """
        prepared_inputs = []
        system_prompt_key = None
        for input in inputs:
            conversation = []
            if "system_prompt" in input:
                conversation.append(
                    {"role": "system", "content": input["system_prompt"]}
                )
            elif self.system_prompt is not None:
                if isinstance(self.system_prompt, list):
                    system_prompt = random.choices(self.system_prompt, k=1)[0]
                elif isinstance(self.system_prompt, dict):
                    system_prompts_keys = list(self.system_prompt.keys())
                    system_prompts_values = list(self.system_prompt.values())
                    weights: Union[List[float], None] = None
                    if isinstance(system_prompts_values[0], tuple):
                        weights = [weight for _, weight in system_prompts_values]  # type: ignore
                    system_prompt_key = random.choices(
                        system_prompts_keys, weights, k=1
                    )[0]
                    system_prompt = self.system_prompt[system_prompt_key]
                    if isinstance(system_prompt, tuple):
                        system_prompt = system_prompt[0]
                else:
                    system_prompt = self.system_prompt
                conversation.append({"role": "system", "content": system_prompt})
            elif self.n_turns > 1:  # type: ignore
                conversation.append(
                    {"role": "system", "content": MAGPIE_MULTI_TURN_SYSTEM_PROMPT}
                )

            prepared_inputs.append(conversation)

        return prepared_inputs, system_prompt_key

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
            if instruction is not None:
                conversation.append({"role": role, "content": instruction})
        return conversations

    def _generate_instruction(
        self, inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        prepared_inputs, system_prompt_key = (
            self._prepare_inputs_for_instruction_generation(inputs)
        )
        outputs = self.llm.generate(
            inputs=prepared_inputs,
            num_generations=1,
            **self.llm.generation_kwargs,  # type: ignore
        )
        rows = []
        for output in outputs:
            row = {"instruction": output[0]}
            if system_prompt_key is not None:
                row["system_prompt_key"] = system_prompt_key
            rows.append(row)
        return rows

    def _prepare_conversation_outputs(
        self, conversations: List["ChatType"], system_prompt_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Prepare the output conversation removing the system prompt if necessary. If
        `n_turns==1`, then it will return a dictionary with "instruction" and "response"
        keys. Otherwise, it will return a dictionary with a "conversation" key.

        Args:
            conversations: the list of generated conversations.
            system_prompt_key: the key of the system prompt used to generate the conversation.

        Returns:
            A list of dictionaries containing a "conversation" key or "instruction" and
            "responses" key.
        """
        outputs = []
        for conversation in conversations:
            # Something went wrong with the `LLM` and it didn't generate any message
            if len(conversation) == 0:
                if self.n_turns == 1:
                    outputs.append({"instruction": None, "response": None})
                else:
                    outputs.append({"conversation": []})
                continue
            if not self.include_system_prompt and conversation[0]["role"] == "system":
                conversation.pop(0)
            if self.n_turns == 1 and len(conversation) == 2:
                output: Dict[str, Any] = {
                    "instruction": conversation[0]["content"],
                    "response": conversation[1]["content"],
                }
            else:
                output = {"conversation": conversation}
            if system_prompt_key is not None:
                output["system_prompt_key"] = system_prompt_key
            outputs.append(output)
        return outputs

    def _generate_conversation_turn(
        self, role: str, conversations: List["ChatType"], active_indices: List[int]
    ) -> Tuple[List["ChatType"], List[int]]:
        # Generate an output for the conversations that are still active (no previous `None`s)
        outputs = self.llm.generate(
            inputs=[conversations[idx] for idx in active_indices],
            num_generations=1,
            **self.llm.generation_kwargs,  # type: ignore
        )

        active_conversations = [conversations[idx] for idx in active_indices]
        updated_conversations = self._append_messages_to_conversations(
            role=role,
            messages=[output[0] for output in outputs],
            conversations=active_conversations,
        )

        for idx, conv in zip(active_indices, updated_conversations):
            conversations[idx] = conv

        new_active_indices = [
            idx for idx, output in zip(active_indices, outputs) if output[0] is not None
        ]

        return conversations, new_active_indices

    def _generate_multi_turn_conversation(
        self, inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        conversations, system_prompt_key = (
            self._prepare_inputs_for_instruction_generation(inputs)
        )
        # Keep track of the active conversations, as it could happen that for some conversation
        # we can't generate the next turn because the `LLM` returned `None`.
        active_indices = list(range(len(conversations)))

        for i in range(self.n_turns):  # type: ignore
            if not active_indices:
                break

            # Generate user message
            conversations, active_indices = self._generate_conversation_turn(
                role="user", conversations=conversations, active_indices=active_indices
            )

            if i == self.n_turns - 1 and self.end_with_user:  # type: ignore
                break

            if not active_indices:
                break

            # Generate assistant message
            conversations, active_indices = self._generate_conversation_turn(
                role="assistant",
                conversations=conversations,
                active_indices=active_indices,
            )

        return self._prepare_conversation_outputs(conversations)

    def _generate_with_pre_query_template(
        self, inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate a list of instructions or conversations of the specified number of turns.

        Args:
            inputs: a list of dictionaries that can contain a `system_prompt` key.

        Returns:
            The list of generated conversations.
        """
        outputs = (
            self._generate_instruction(inputs)
            if self.only_instruction
            else self._generate_multi_turn_conversation(inputs)
        )

        return [
            {**input, **output, "model_name": self.llm.model_name}
            for input, output in zip(inputs, outputs)
        ]


class Magpie(MagpieBase, Task):
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

    Runtime parameters:
        - `n_turns`: the number of turns that the generated conversation will have. Defaults
            to `1`.
        - `end_with_user`: whether the conversation should end with a user message.
            Defaults to `False`.
        - `include_system_prompt`: whether to include the system prompt used in the generated
            conversation. Defaults to `False`.
        - `only_instruction`: whether to generate only the instruction. If this argument is
            `True`, then `n_turns` will be ignored. Defaults to `False`.
        - `system_prompt`: an optional system prompt or list of system prompts that can
            be used to steer the LLM to generate content of certain topic, guide the style,
            etc. If it's a list of system prompts, then a random system prompt will be chosen
            per input/output batch. If the provided inputs contains a `system_prompt` column,
            then this runtime parameter will be ignored and the one from the column will
            be used. Defaults to `None`.
        - `system_prompt`: an optional system prompt, or a list of system prompts from which
            a random one will be chosen, or a dictionary of system prompts from which a
            random one will be choosen, or a dictionary of system prompts with their probability
            of being chosen. The random system prompt will be chosen per input/output batch.
            This system prompt can be used to guide the generation of the instruct LLM and
            steer it to generate instructions of a certain topic.

    Input columns:
        - system_prompt (`str`, optional): an optional system prompt that can be provided
            to guide the generation of the instruct LLM and steer it to generate instructions
            of certain topic.

    Output columns:
        - conversation (`ChatType`): the generated conversation which is a list of chat
            items with a role and a message. Only if `only_instruction=False`.
        - instruction (`str`): the generated instructions if `only_instruction=True` or `n_turns==1`.
        - response (`str`): the generated response if `n_turns==1`.
        - system_prompt_key (`str`, optional): the key of the system prompt used to generate
            the conversation or instruction. Only if `system_prompt` is a dictionary.
        - model_name (`str`): The model name used to generate the `conversation` or `instruction`.

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
            raise DistilabelUserError(
                f"`Magpie` task can only be used with an `LLM` that uses the `MagpieChatTemplateMixin`."
                f"`{self.llm.__class__.__name__}` doesn't use the aforementioned mixin.",
                page="components-gallery/tasks/magpie/",
            )

        self.llm.use_magpie_template = True

    @property
    def inputs(self) -> "StepColumns":
        return {"system_prompt": False}

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """Does nothing."""
        return []

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
