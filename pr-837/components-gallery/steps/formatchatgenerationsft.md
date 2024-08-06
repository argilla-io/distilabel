---
hide:
  - navigation
---
# FormatChatGenerationSFT

Format the output of a `ChatGeneration` task for Supervised Fine-Tuning (SFT) following the



standard formatting from frameworks such as `axolotl` or `alignment-handbook`.

    `FormatChatGenerationSFT` is a `Step` that formats the output of a `ChatGeneration` task for
    Supervised Fine-Tuning (SFT) following the standard formatting from frameworks such as `axolotl`
    or `alignment-handbook`. The output of the `ChatGeneration` task is formatted into a chat-like
    conversation with the `instruction` as the user message and the `generation` as the assistant
    message. Optionally, if the `system_prompt` is available, it is included as the first message
    in the conversation.








### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[generation]
		end
		subgraph New columns
			OCOL0[prompt]
			OCOL1[prompt_id]
			OCOL2[messages]
		end
	end

	subgraph FormatChatGenerationSFT
		StepInput[Input Columns: instruction, generation]
		StepOutput[Output Columns: prompt, prompt_id, messages]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The instruction used to generate the `generation` with the `LLM`.

- **generation** (`str`): The generation produced by the `LLM`.




#### Outputs


- **prompt** (`str`): The instruction used to generate the `generation` with the `LLM`.

- **prompt_id** (`str`): The `SHA256` hash of the `prompt`.

- **messages** (`List[Dict[str, str]]`): The chat-like conversation with the `instruction` as  the user message and the `generation` as the assistant message.







