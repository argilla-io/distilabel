---
hide:
  - navigation
---
# Genstruct

Generate a pair of instruction-response from a document using an `LLM`.



`Genstruct` is a pre-defined task designed to generate valid instructions from a given raw document,
    with the title and the content, enabling the creation of new, partially synthetic instruction finetuning
    datasets from any raw-text corpus. The task is based on the Genstruct 7B model by Nous Research, which is
    inspired in the Ada-Instruct paper.



### Note
The Genstruct prompt i.e. the task, can be used with any model really, but the safest / recommended
option is to use `NousResearch/Genstruct-7B` as the LLM provided to the task, since it was trained
for this specific task.



### Attributes

- **_template**: a Jinja2 template used to format the input for the LLM.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[title]
			ICOL1[content]
		end
		subgraph New columns
			OCOL0[user]
			OCOL1[assistant]
			OCOL2[model_name]
		end
	end

	subgraph Genstruct
		StepInput[Input Columns: title, content]
		StepOutput[Output Columns: user, assistant, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **title** (`str`): The title of the document.

- **content** (`str`): The content of the document.




#### Outputs


- **user** (`str`): The user's instruction based on the document.

- **assistant** (`str`): The assistant's response based on the user's instruction.

- **model_name** (`str`): The model name used to generate the `feedback` and `result`.





### Examples


#### Generate instructions from raw documents using the title and content
```python
from distilabel.steps.tasks import Genstruct
from distilabel.llms.huggingface import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
genstruct = Genstruct(
    llm=InferenceEndpointsLLM(
        model_id="NousResearch/Genstruct-7B",
    ),
)

genstruct.load()

result = next(
    genstruct.process(
        [
            {"title": "common instruction", "content": "content of the document"},
        ]
    )
)
# result
# [
#     {
#         'title': 'An instruction',
#         'content': 'content of the document',
#         'model_name': 'test',
#         'user': 'An instruction',
#         'assistant': 'content of the document',
#     }
# ]
```




### References

- [Genstruct 7B by Nous Research](https://huggingface.co/NousResearch/Genstruct-7B)

- [Ada-Instruct: Adapting Instruction Generators for Complex Reasoning](https://arxiv.org/abs/2310.04484)


