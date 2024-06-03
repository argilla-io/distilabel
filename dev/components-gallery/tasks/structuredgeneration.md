# StructuredGeneration


Generate structured content for a given `instruction` using an `LLM`.



`StructuredGeneration` is a pre-defined task that defines the `instruction` and the `grammar`
    as the inputs, and `generation` as the output. This task is used to generate structured content based on
    the input instruction and following the schema provided within the `grammar` column per each
    `instruction`. The `model_name` also returned as part of the output in order to enhance it.





### Attributes

- **use_system_prompt**: Whether to use the system prompt in the generation. Defaults to `True`,  which means that if the column `system_prompt` is defined within the input batch, then  the `system_prompt` will be used, otherwise, it will be ignored.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[grammar]
		end
		subgraph New columns
			OCOL0[generation]
			OCOL1[model_name]
		end
	end

	subgraph StructuredGeneration
		StepInput[Input Columns: instruction, grammar]
		StepOutput[Output Columns: generation, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The instruction to generate structured content from.

- **grammar** (`Dict[str, Any]`): The grammar to generate structured content from. It should be a  Python dictionary with the keys `type` and `value`, where `type` should be one of `json` or  `regex`, and the `value` should be either the JSON schema or the regex pattern, respectively.




#### Outputs


- **generation** (`str`): The generated text matching the provided schema, if possible.

- **model_name** (`str`): The name of the model used to generate the text.







