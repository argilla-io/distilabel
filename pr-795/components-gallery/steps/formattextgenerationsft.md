---
hide:
  - navigation
---
# FormatTextGenerationSFT

Format the output of a `TextGeneration` task for Supervised Fine-Tuning (SFT).



`FormatTextGenerationSFT` is a `Step` that formats the output of a `TextGeneration` task for
    Supervised Fine-Tuning (SFT) following the standard formatting from frameworks such as `axolotl`
    or `alignment-handbook`. The output of the `TextGeneration` task is formatted into a chat-like
    conversation with the `instruction` as the user message and the `generation` as the assistant
    message. Optionally, if the `system_prompt` is available, it is included as the first message
    in the conversation.








### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[system_prompt]
			ICOL1[instruction]
			ICOL2[generation]
		end
		subgraph New columns
			OCOL0[prompt]
			OCOL1[prompt_id]
			OCOL2[messages]
		end
	end

	subgraph FormatTextGenerationSFT
		StepInput[Input Columns: system_prompt, instruction, generation]
		StepOutput[Output Columns: prompt, prompt_id, messages]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **system_prompt** (`str`, optional): The system prompt used within the `LLM` to generate the  `generation`, if available.

- **instruction** (`str`): The instruction used to generate the `generation` with the `LLM`.

- **generation** (`str`): The generation produced by the `LLM`.




#### Outputs


- **prompt** (`str`): The instruction used to generate the `generation` with the `LLM`.

- **prompt_id** (`str`): The `SHA256` hash of the `prompt`.

- **messages** (`List[Dict[str, str]]`): The chat-like conversation with the `instruction` as  the user message and the `generation` as the assistant message.





### Examples


#### Format your dataset for SFT fine tuning
```python
from distilabel.steps import FormatTextGenerationSFT

format_sft = FormatTextGenerationSFT()
format_sft.load()

# NOTE: "system_prompt" can be added optionally.
result = next(
    format_sft.process(
        [
            {
                "instruction": "What's 2+2?",
                "generation": "4"
            }
        ]
    )
)
# >>> result
# [
#     {
#         'instruction': 'What's 2+2?',
#         'generation': '4',
#         'prompt': 'What's 2+2?',
#         'prompt_id': '7762ecf17ad41479767061a8f4a7bfa3b63d371672af5180872f9b82b4cd4e29',
#         'messages': [{'role': 'user', 'content': "What's 2+2?"}, {'role': 'assistant', 'content': '4'}]
#     }
# ]
```




