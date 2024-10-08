---
hide:
  - navigation
---
# Magpie

Generates conversations using an instruct fine-tuned LLM.



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





### Attributes

- **n_turns**: the number of turns that the generated conversation will have.  Defaults to `1`.

- **end_with_user**: whether the conversation should end with a user message.  Defaults to `False`.

- **include_system_prompt**: whether to include the system prompt used in the generated  conversation. Defaults to `False`.

- **only_instruction**: whether to generate only the instruction. If this argument is  `True`, then `n_turns` will be ignored. Defaults to `False`.

- **system_prompt**: an optional system prompt, or a list of system prompts from which  a random one will be chosen, or a dictionary of system prompts from which a  random one will be choosen, or a dictionary of system prompts with their probability  of being chosen. The random system prompt will be chosen per input/output batch.  This system prompt can be used to guide the generation of the instruct LLM and  steer it to generate instructions of a certain topic. Defaults to `None`.




### Runtime Parameters

- **n_turns**: the number of turns that the generated conversation will have. Defaults  to `1`.

- **end_with_user**: whether the conversation should end with a user message.  Defaults to `False`.

- **include_system_prompt**: whether to include the system prompt used in the generated  conversation. Defaults to `False`.

- **only_instruction**: whether to generate only the instruction. If this argument is  `True`, then `n_turns` will be ignored. Defaults to `False`.

- **system_prompt**: an optional system prompt, or a list of system prompts from which  a random one will be chosen, or a dictionary of system prompts from which a  random one will be choosen, or a dictionary of system prompts with their probability  of being chosen. The random system prompt will be chosen per input/output batch.  This system prompt can be used to guide the generation of the instruct LLM and  steer it to generate instructions of a certain topic.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[system_prompt]
		end
		subgraph New columns
			OCOL0[conversation]
			OCOL1[instruction]
			OCOL2[response]
			OCOL3[system_prompt_key]
			OCOL4[model_name]
		end
	end

	subgraph Magpie
		StepInput[Input Columns: system_prompt]
		StepOutput[Output Columns: conversation, instruction, response, system_prompt_key, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepOutput --> OCOL3
	StepOutput --> OCOL4
	StepInput --> StepOutput

```


#### Inputs


- **system_prompt** (`str`, optional): an optional system prompt that can be provided  to guide the generation of the instruct LLM and steer it to generate instructions  of certain topic.




#### Outputs


- **conversation** (`ChatType`): the generated conversation which is a list of chat  items with a role and a message. Only if `only_instruction=False`.

- **instruction** (`str`): the generated instructions if `only_instruction=True` or `n_turns==1`.

- **response** (`str`): the generated response if `n_turns==1`.

- **system_prompt_key** (`str`, optional): the key of the system prompt used to generate  the conversation or instruction. Only if `system_prompt` is a dictionary.

- **model_name** (`str`): The model name used to generate the `conversation` or `instruction`.





### Examples


#### Generating instructions with Llama 3 8B Instruct and TransformersLLM
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

#### Generating conversations with Llama 3 8B Instruct and TransformersLLM
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
#                 'content': 'I'm having trouble solving the limits of functions in calculus. Could you explain how to work with them? Limits of functions are denoted by lim x→a f(x) or lim x→a [f(x)]. It is read as "the limit as x approaches a of f
# of x".'
#             },
#             {
#                 'role': 'assistant',
#                 'content': 'Limits are indeed a fundamental concept in calculus, and understanding them can be a bit tricky at first, but don't worry, I'm here to help! The notation lim x→a f(x) indeed means "the limit as x approaches a of f of
# x". What it's asking us to do is find the'
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




### References

- [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)


