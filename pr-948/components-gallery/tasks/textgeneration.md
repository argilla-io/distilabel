---
hide:
  - navigation
---
# TextGeneration

Simple text generation with an `LLM` given an instruction.



`TextGeneration` is a pre-defined task that defines the `instruction` as the input
    and `generation` as the output. This task is used to generate text based on the input
    instruction. The model_name is also returned as part of the output in order to enhance it.





### Attributes

- **system_prompt**: The system prompt to use in the generation. If not provided, then  it will check if the input row has a column named `system_prompt` and use it.  If not, then no system prompt will be used. Defaults to `None`.

- **use_system_prompt**: DEPRECATED. To be removed in 1.5.0. Whether to use the system  prompt in the generation. Defaults to `True`, which means that if the column  `system_prompt` is defined within the input batch, then the `system_prompt`  will be used, otherwise, it will be ignored.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
		end
		subgraph New columns
			OCOL0[generation]
			OCOL1[model_name]
		end
	end

	subgraph TextGeneration
		StepInput[Input Columns: instruction]
		StepOutput[Output Columns: generation, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The instruction to generate text from.




#### Outputs


- **generation** (`str`): The generated text.

- **model_name** (`str`): The name of the model used to generate the text.





### Examples


#### Generate text from an instruction
```python
from distilabel.steps.tasks import TextGeneration
from distilabel.llms.huggingface import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
text_gen = TextGeneration(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    )
)

text_gen.load()

result = next(
    text_gen.process(
        [{"instruction": "your instruction"}]
    )
)
# result
# [
#     {
#         'instruction': 'your instruction',
#         'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
#         'generation': 'generation',
#     }
# ]
```




