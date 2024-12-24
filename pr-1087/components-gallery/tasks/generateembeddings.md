---
hide:
  - navigation
---
# GenerateEmbeddings

Generate embeddings using the last hidden state of an `LLM`.



Generate embeddings for a text input using the last hidden state of an `LLM`, as
    described in the paper 'What Makes Good Data for Alignment? A Comprehensive Study of
    Automatic Data Selection in Instruction Tuning'.





### Attributes

- **llm**: The `LLM` to use to generate the embeddings.

- **system_prompt**: The system prompt for the embedding generation task.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[text]
			ICOL1[system_prompt]
		end
		subgraph New columns
			OCOL0[embedding]
			OCOL1[model_name]
		end
	end

	subgraph GenerateEmbeddings
		StepInput[Input Columns: text, system_prompt]
		StepOutput[Output Columns: embedding, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **text** (`str`, `List[Dict[str, str]]`): The input text or conversation to generate  embeddings for.

- **system_prompt** (`Optional[str]`): The system prompt for the embedding generation task.




#### Outputs


- **embedding** (`List[float]`): The embedding of the input text or conversation.

- **model_name** (`str`): The model name used to generate the embeddings.





### Examples


#### Rank LLM candidates
```python
from distilabel.steps.tasks import GenerateEmbeddings
from distilabel.models.llms.huggingface import TransformersLLM

# Consider this as a placeholder for your actual LLM.
embedder = GenerateEmbeddings(
    llm=TransformersLLM(
        model="TaylorAI/bge-micro-v2",
        model_kwargs={"is_decoder": True},
        cuda_devices=[],
    )
)
embedder.load()

result = next(
    embedder.process(
        [
            {"text": "Hello, how are you?"},
        ]
    )
)
```




### References

- [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685)


