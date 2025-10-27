---
hide:
  - navigation
---
# ConversationTemplate

Generate a conversation template from an instruction and a response.










### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[response]
		end
		subgraph New columns
			OCOL0[conversation]
		end
	end

	subgraph ConversationTemplate
		StepInput[Input Columns: instruction, response]
		StepOutput[Output Columns: conversation]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The instruction to be used in the conversation.

- **response** (`str`): The response to be used in the conversation.




#### Outputs


- **conversation** (`ChatType`): The conversation template.





### Examples


#### Create a conversation from an instruction and a response
```python
from distilabel.steps import ConversationTemplate

conv_template = ConversationTemplate()
conv_template.load()

result = next(
    conv_template.process(
        [
            {
                "instruction": "Hello",
                "response": "Hi",
            }
        ],
    )
)
# >>> result
# [{'instruction': 'Hello', 'response': 'Hi', 'conversation': [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}]}]
```




