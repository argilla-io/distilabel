import os
from distilabel.tasks import EvolInstructTask
from distilabel.llm import InferenceEndpointsLLM

task = EvolInstructTask()

llm = InferenceEndpointsLLM(
    task=task,
    endpoint_name_or_model_id="aws-notus-7b-v1-3184",
    endpoint_namespace="argilla",
    token=os.getenv("HF_API_TOKEN", None),
    prompt_format="notus"
)
print(llm.validate_prompts([{"input": "What's a large language model?"}])[0])
# <|system|>
# </s>
# <|user|>
# I want you to act as a Prompt Rewriter.
# Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.
# But the rewritten prompt must be reasonable and must be understood and responded by humans.
# Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.
# You SHOULD complicate the given prompt using the following method:
# Please add one more constraints/requirements into #The Given Prompt#
# You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.
# '#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
# #The Given Prompt#:
# What's a large language model?

# #Rewritten Prompt#:
# </s>
# <|assistant|>