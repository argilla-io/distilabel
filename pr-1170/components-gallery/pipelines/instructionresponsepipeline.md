---
hide:
  - navigation
---
# InstructionResponsePipeline

Generates instructions and responses for a given system prompt.



This example pipeline can be used for a Supervised Fine-Tuning dataset which you
    could use to train or evaluate a model. The pipeline generates instructions using the
    MagpieGenerator and responses for a given system prompt. The pipeline then keeps only
    the instruction, response, and model_name columns.





### Attributes

- **llm**: The LLM to use for generating instructions and responses. Defaults to  InferenceEndpointsLLM with Meta-Llama-3.1-8B-Instruct.

- **system_prompt**: The system prompt to use for generating instructions and responses.  Defaults to "You are a creative AI Assistant writer."

- **hf_token**: The Hugging Face token to use for accessing the model. Defaults to None.

- **n_turns**: The number of turns to generate for each conversation. Defaults to 1.

- **num_rows**: The number of rows to generate. Defaults to 10.

- **batch_size**: The batch size to use for generation. Defaults to 1.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph New columns
			OCOL0[conversation]
			OCOL1[instruction]
			OCOL2[response]
			OCOL3[system_prompt_key]
			OCOL4[model_name]
		end
	end

	subgraph InstructionResponsePipeline
		StepOutput[Output Columns: conversation, instruction, response, system_prompt_key, model_name]
	end

	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepOutput --> OCOL3
	StepOutput --> OCOL4

```




#### Outputs


- **conversation** (`ChatType`): the generated conversation which is a list of chat  items with a role and a message.

- **instruction** (`str`): the generated instructions if `only_instruction=True`.

- **response** (`str`): the generated response if `n_turns==1`.

- **system_prompt_key** (`str`, optional): the key of the system prompt used to generate  the conversation or instruction. Only if `system_prompt` is a dictionary.

- **model_name** (`str`): The model name used to generate the `conversation` or `instruction`.





### Examples


#### Generate instructions and responses for a given system prompt
```python
from distilabel.pipeline import InstructionResponsePipeline

pipeline = InstructionResponsePipeline()

distiset = pipeline.run()
```

#### Customizing the pipeline further
```python
from distilabel.pipeline import InstructionResponsePipeline

pipeline = InstructionResponsePipeline(
    system_prompt="You are a creative AI Assistant for writing science fiction.",
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.2-3B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3.2-3B-Instruct",
        generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
    ),
    num_rows=500,
    batch_size=2,
    n_turns=2,
)
```




### References

- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)


