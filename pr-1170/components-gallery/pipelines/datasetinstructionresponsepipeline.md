---
hide:
  - navigation
---
# DatasetInstructionResponsePipeline

Generates instructions and responses for a dataset with input documents.



This example pipeline can be used for a Supervised Fine-Tuning dataset which you
    could use to train or evaluate a model. The pipeline generates instructions using the
    SelfInstruct step and TextGeneration step.





### Attributes

- **llm**: The LLM to use for generating instructions and responses. Defaults to  InferenceEndpointsLLM with Meta-Llama-3.1-8B-Instruct.

- **system_prompt**: The system prompt to use for generating instructions and responses.  Defaults to "You are a creative AI Assistant writer."

- **hf_token**: The Hugging Face token to use for accessing the model. Defaults to None.

- **num_instructions**: The number of instructions to generate. Defaults to 2.

- **batch_size**: The batch size to use for generation. Defaults to 1.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[input]
		end
		subgraph New columns
			OCOL0[conversation]
			OCOL1[instruction]
			OCOL2[response]
			OCOL3[system_prompt_key]
			OCOL4[model_name]
		end
	end

	subgraph DatasetInstructionResponsePipeline
		StepInput[Input Columns: input]
		StepOutput[Output Columns: conversation, instruction, response, system_prompt_key, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepOutput --> OCOL3
	StepOutput --> OCOL4
	StepInput --> StepOutput

```


#### Inputs


- **input** (`str`): The input document to generate instructions and responses for.




#### Outputs


- **conversation** (`ChatType`): the generated conversation which is a list of chat  items with a role and a message.

- **instruction** (`str`): the generated instructions if `only_instruction=True`.

- **response** (`str`): the generated response if `n_turns==1`.

- **system_prompt_key** (`str`, optional): the key of the system prompt used to generate  the conversation or instruction. Only if `system_prompt` is a dictionary.

- **model_name** (`str`): The model name used to generate the `conversation` or `instruction`.





### Examples


#### Generate instructions and responses for a given system prompt
```python
from datasets import Dataset
from distilabel.pipeline import DatasetInstructionResponsePipeline

pipeline = DatasetInstructionResponsePipeline(num_instructions=5)

distiset = pipeline.run(
    use_cache=False,
    dataset=Dataset.from_list(
        mapping=[
            {
                "input": "<document>",
            }
        ]
    ),
)
```




### References

- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)


