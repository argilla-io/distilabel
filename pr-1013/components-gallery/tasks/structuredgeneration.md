---
hide:
  - navigation
---
# StructuredGeneration

Generate structured content for a given `instruction` using an `LLM`.



`StructuredGeneration` is a pre-defined task that defines the `instruction` and the `structured_output`
    as the inputs, and `generation` as the output. This task is used to generate structured content based on
    the input instruction and following the schema provided within the `structured_output` column per each
    `instruction`. The `model_name` also returned as part of the output in order to enhance it.





### Attributes

- **use_system_prompt**: Whether to use the system prompt in the generation. Defaults to `True`,  which means that if the column `system_prompt` is defined within the input batch, then  the `system_prompt` will be used, otherwise, it will be ignored.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[structured_output]
		end
		subgraph New columns
			OCOL0[generation]
			OCOL1[model_name]
		end
	end

	subgraph StructuredGeneration
		StepInput[Input Columns: instruction, structured_output]
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

- **structured_output** (`Dict[str, Any]`): The structured_output to generate structured content from. It should be a  Python dictionary with the keys `format` and `schema`, where `format` should be one of `json` or  `regex`, and the `schema` should be either the JSON schema or the regex pattern, respectively.




#### Outputs


- **generation** (`str`): The generated text matching the provided schema, if possible.

- **model_name** (`str`): The name of the model used to generate the text.





### Examples


#### Generate structured output from a JSON schema
```python
from distilabel.steps.tasks import StructuredGeneration
from distilabel.llms import InferenceEndpointsLLM

structured_gen = StructuredGeneration(
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
    ),
)

structured_gen.load()

result = next(
    structured_gen.process(
        [
            {
                "instruction": "Create an RPG character",
                "structured_output": {
                    "format": "json",
                    "schema": {
                        "properties": {
                            "name": {
                                "title": "Name",
                                "type": "string"
                            },
                            "description": {
                                "title": "Description",
                                "type": "string"
                            },
                            "role": {
                                "title": "Role",
                                "type": "string"
                            },
                            "weapon": {
                                "title": "Weapon",
                                "type": "string"
                            }
                        },
                        "required": [
                            "name",
                            "description",
                            "role",
                            "weapon"
                        ],
                        "title": "Character",
                        "type": "object"
                    }
                },
            }
        ]
    )
)
```

#### Generate structured output from a regex pattern (only works with LLMs that support regex, the providers using outlines)
```python
from distilabel.steps.tasks import StructuredGeneration
from distilabel.llms import InferenceEndpointsLLM

structured_gen = StructuredGeneration(
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
    ),
)

structured_gen.load()

result = next(
    structured_gen.process(
        [
            {
                "instruction": "What's the weather like today in Seattle in Celsius degrees?",
                "structured_output": {
                    "format": "regex",
                    "schema": r"(\d{1,2})°C"
                },

            }
        ]
    )
)
```




