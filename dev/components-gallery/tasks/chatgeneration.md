# ChatGeneration


Generates text based on a conversation.



`ChatGeneration` is a pre-defined task that defines the `messages` as the input
    and `generation` as the output. This task is used to generate text based on a conversation.
    The `model_name` is also returned as part of the output in order to enhance it.








### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[messages]
		end
		subgraph New columns
			OCOL0[generation]
			OCOL1[model_name]
		end
	end

	subgraph ChatGeneration
		StepInput[Input Columns: messages]
		StepOutput[Output Columns: generation, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **messages** (`List[Dict[Literal["role", "content"], str]]`): The messages to generate the  follow up completion from.




#### Outputs


- **generation** (`str`): The generated text from the assistant.

- **model_name** (`str`): The model name used to generate the text.







