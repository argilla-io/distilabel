---
hide:
  - navigation
---
# SelfInstruct

Generate instructions based on a given input using an `LLM`.



`SelfInstruct` is a pre-defined task that, given a number of instructions, a
    certain criteria for query generations, an application description, and an input,
    generates a number of instruction related to the given input and following what
    is stated in the criteria for query generation and the application description.
    It is based in the SelfInstruct framework from the paper "Self-Instruct: Aligning
    Language Models with Self-Generated Instructions".





### Attributes

- **num_instructions**: The number of instructions to be generated. Defaults to 5.

- **criteria_for_query_generation**: The criteria for the query generation. Defaults  to the criteria defined within the paper.

- **application_description**: The description of the AI application that one want  to build with these instructions. Defaults to `AI assistant`.

- **system_prompt**: The system prompt for the SelfInstruct task.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[input]
			ICOL1[system_prompt]
		end
		subgraph New columns
			OCOL0[instructions]
			OCOL1[model_name]
		end
	end

	subgraph SelfInstruct
		StepInput[Input Columns: input, system_prompt]
		StepOutput[Output Columns: instructions, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **input** (`str`): The input to generate the instructions. It's also called seed in  the paper.

- **system_prompt** (`Optional[str]`): The system prompt for the SelfInstruct task.




#### Outputs


- **instructions** (`List[str]`): The generated instructions.

- **model_name** (`str`): The model name used to generate the instructions.





### Examples


#### Generate instructions based on a given input
```python
from distilabel.steps.tasks import SelfInstruct
from distilabel.models import InferenceEndpointsLLM

self_instruct = SelfInstruct(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    num_instructions=5,  # This is the default value
)

self_instruct.load()

result = next(self_instruct.process([{"input": "instruction"}]))
# result
# [
#     {
#         'input': 'instruction',
#         'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
#         'instructions': ["instruction 1", "instruction 2", "instruction 3", "instruction 4", "instruction 5"],
#     }
# ]
```




