---
hide:
  - navigation
---
# URIAL

Generates a response using a non-instruct fine-tuned model.



`URIAL` is a pre-defined task that generates a response using a non-instruct fine-tuned
    model. This task is used to generate a response based on the conversation provided as
    input.








### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[conversation]
		end
		subgraph New columns
			OCOL0[generation]
			OCOL1[model_name]
		end
	end

	subgraph URIAL
		StepInput[Input Columns: instruction, conversation]
		StepOutput[Output Columns: generation, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`, optional): The instruction to generate a response from.

- **conversation** (`List[Dict[str, str]]`, optional): The conversation to generate  a response from (the last message must be from the user).




#### Outputs


- **generation** (`str`): The generated response.

- **model_name** (`str`): The name of the model used to generate the response.





### Examples


#### Generate text from an instruction
```python
from distilabel.models import vLLM
from distilabel.steps.tasks import URIAL

step = URIAL(
    llm=vLLM(
        model="meta-llama/Meta-Llama-3.1-8B",
        generation_kwargs={"temperature": 0.7},
    ),
)

step.load()

results = next(
    step.process(inputs=[{"instruction": "What's the most most common type of cloud?"}])
)
# [
#     {
#         'instruction': "What's the most most common type of cloud?",
#         'generation': 'Clouds are classified into three main types, high, middle, and low. The most common type of cloud is the middle cloud.',
#         'distilabel_metadata': {...},
#         'model_name': 'meta-llama/Meta-Llama-3.1-8B'
#     }
# ]
```




### References

- [The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning](https://arxiv.org/abs/2312.01552)


