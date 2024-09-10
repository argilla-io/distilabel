---
hide:
  - navigation
---
# RewardModelScore

Assign a score to a response using a Reward Model.



`RewardModelScore` is a `Step` that using a Reward Model (RM) loaded using `transformers`,
    assigns an score to a response generated for an instruction, or a score to a multi-turn
    conversation.





### Attributes

- **model**: the model Hugging Face Hub repo id or a path to a directory containing the  model weights and configuration files.

- **revision**: if `model` refers to a Hugging Face Hub repository, then the revision  (e.g. a branch name or a commit id) to use. Defaults to `"main"`.

- **torch_dtype**: the torch dtype to use for the model e.g. "float16", "float32", etc.  Defaults to `"auto"`.

- **trust_remote_code**: whether to allow fetching and executing remote code fetched  from the repository in the Hub. Defaults to `False`.

- **device_map**: a dictionary mapping each layer of the model to a device, or a mode like `"sequential"` or `"auto"`. Defaults to `None`.

- **token**: the Hugging Face Hub token that will be used to authenticate to the Hugging  Face Hub. If not provided, the `HF_TOKEN` environment or `huggingface_hub` package  local configuration will be used. Defaults to `None`.

- **truncation**: whether to truncate sequences at the maximum length. Defaults to `False`.

- **max_length**: maximun length to use for padding or truncation. Defaults to `None`.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[response]
			ICOL2[conversation]
		end
		subgraph New columns
			OCOL0[score]
		end
	end

	subgraph RewardModelScore
		StepInput[Input Columns: instruction, response, conversation]
		StepOutput[Output Columns: score]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`, optional): the instruction used to generate a `response`.  If provided, then `response` must be provided too.

- **response** (`str`, optional): the response generated for `instruction`. If provided,  then `instruction` must be provide too.

- **conversation** (`ChatType`, optional): a multi-turn conversation. If not provided,  then `instruction` and `response` columns must be provided.




#### Outputs


- **score** (`float`): the score given by the reward model for the instruction-response  pair or the conversation.





### Examples


#### response pair
```python
from distilabel.steps import RewardModelScore

step = RewardModelScore(
    model="RLHFlow/ArmoRM-Llama3-8B-v0.1", device_map="auto", trust_remote_code=True
)

step.load()

result = next(
    step.process(
        inputs=[
            {
                "instruction": "How much is 2+2?",
                "response": "The output of 2+2 is 4",
            },
            {"instruction": "How much is 2+2?", "response": "4"},
        ]
    )
)
# [
#   {'instruction': 'How much is 2+2?', 'response': 'The output of 2+2 is 4', 'score': 0.11690367758274078},
#   {'instruction': 'How much is 2+2?', 'response': '4', 'score': 0.10300665348768234}
# ]
```

#### turn conversation
```python
from distilabel.steps import RewardModelScore

step = RewardModelScore(
    model="RLHFlow/ArmoRM-Llama3-8B-v0.1", device_map="auto", trust_remote_code=True
)

step.load()

result = next(
    step.process(
        inputs=[
            {
                "conversation": [
                    {"role": "user", "content": "How much is 2+2?"},
                    {"role": "assistant", "content": "The output of 2+2 is 4"},
                ],
            },
            {
                "conversation": [
                    {"role": "user", "content": "How much is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
            },
        ]
    )
)
# [
#   {'conversation': [{'role': 'user', 'content': 'How much is 2+2?'}, {'role': 'assistant', 'content': 'The output of 2+2 is 4'}], 'score': 0.11690367758274078},
#   {'conversation': [{'role': 'user', 'content': 'How much is 2+2?'}, {'role': 'assistant', 'content': '4'}], 'score': 0.10300665348768234}
# ]
```




