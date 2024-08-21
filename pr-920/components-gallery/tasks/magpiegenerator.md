---
hide:
  - navigation
---
# MagpieGenerator

Generator task the generates instructions or conversations using Magpie.



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





### Attributes

- **n_turns**: the number of turns that the generated conversation will have.  Defaults to `1`.

- **end_with_user**: whether the conversation should end with a user message.  Defaults to `False`.

- **include_system_prompt**: whether to include the system prompt used in the generated  conversation. Defaults to `False`.

- **only_instruction**: whether to generate only the instruction. If this argument is  `True`, then `n_turns` will be ignored. Defaults to `False`.

- **system_prompt**: an optional system prompt or list of system prompts that can  be used to steer the LLM to generate content of certain topic, guide the style,  etc. If it's a list of system prompts, then a random system prompt will be chosen  per input/output batch. If the provided inputs contains a `system_prompt` column,  then this runtime parameter will be ignored and the one from the column will  be used. Defaults to `None`.

- **num_rows**: the number of rows to be generated.




### Runtime Parameters

- **n_turns**: the number of turns that the generated conversation will have. Defaults  to `1`.

- **end_with_user**: whether the conversation should end with a user message.  Defaults to `False`.

- **include_system_prompt**: whether to include the system prompt used in the generated  conversation. Defaults to `False`.

- **only_instruction**: whether to generate only the instruction. If this argument is  `True`, then `n_turns` will be ignored. Defaults to `False`.

- **system_prompt**: an optional system prompt or list of system prompts that can  be used to steer the LLM to generate content of certain topic, guide the style,  etc. If it's a list of system prompts, then a random system prompt will be chosen  per input/output batch. If the provided inputs contains a `system_prompt` column,  then this runtime parameter will be ignored and the one from the column will  be used. Defaults to `None`.

- **num_rows**: the number of rows to be generated.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph New columns
			OCOL0[conversation]
			OCOL1[instruction]
			OCOL2[response]
			OCOL3[model_name]
		end
	end

	subgraph MagpieGenerator
		StepOutput[Output Columns: conversation, instruction, response, model_name]
	end

	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepOutput --> OCOL3

```




#### Outputs


- **conversation** (`ChatType`): the generated conversation which is a list of chat  items with a role and a message.

- **instruction** (`str`): the generated instructions if `only_instruction=True`.

- **response** (`str`): the generated response if `n_turns==1`.

- **model_name** (`str`): The model name used to generate the `conversation` or `instruction`.





### Examples


#### Generating instructions with Llama 3 8B Instruct and TransformersLLM
```python
from distilabel.llms import TransformersLLM
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

#### Generating a conversation with Llama 3 8B Instruct and TransformersLLM
```python
from distilabel.llms import TransformersLLM
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
#                     'content': 'Let me stop you there. Let's explore this "purpose" factor that you mentioned earlier. Can you elaborate more on what type of devices would be suitable for different purposes? For example, if I're primarily using my
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




### References

- [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)


