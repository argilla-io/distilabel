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
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
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
#         'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
#         'generation': 'generation',
#     }
# ]
```

#### Use a custom template to generate text
```python
from distilabel.steps.tasks import TextGeneration
from distilabel.llms.huggingface import InferenceEndpointsLLM

CUSTOM_TEMPLATE = '''        Document:
{{ document }}

Question: {{ question }}

Please provide a clear and concise answer to the question based on the information in the document and your general knowledge:
'''.rstrip()

text_gen = TextGeneration(
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    system_prompt="You are a helpful AI assistant. Your task is to answer the following question based on the provided document. If the answer is not explicitly stated in the document, use your knowledge to provide the most relevant and accurate answer possible. If you cannot answer the question based on the given information, state that clearly.",
    template=CUSTOM_TEMPLATE,
    columns=["document", "question"],
)

text_gen.load()

result = next(
    text_gen.process(
        [
            {
                "document": "The Great Barrier Reef, located off the coast of Australia, is the world's largest coral reef system. It stretches over 2,300 kilometers and is home to a diverse array of marine life, including over 1,500 species of fish. However, in recent years, the reef has faced significant challenges due to climate change, with rising sea temperatures causing coral bleaching events.",
                "question": "What is the main threat to the Great Barrier Reef mentioned in the document?"
            }
        ]
    )
)
# result
# [
#     {
#         'document': 'The Great Barrier Reef, located off the coast of Australia, is the world's largest coral reef system. It stretches over 2,300 kilometers and is home to a diverse array of marine life, including over 1,500 species of fish. However, in recent years, the reef has faced significant challenges due to climate change, with rising sea temperatures causing coral bleaching events.',
#         'question': 'What is the main threat to the Great Barrier Reef mentioned in the document?',
#         'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
#         'generation': 'According to the document, the main threat to the Great Barrier Reef is climate change, specifically rising sea temperatures causing coral bleaching events.',
#     }
# ]
```

#### shot learning with different system prompts
```python
from distilabel.steps.tasks import TextGeneration
from distilabel.llms.huggingface import InferenceEndpointsLLM

CUSTOM_TEMPLATE = '''        Generate a clear, single-sentence instruction based on the following examples:

{% for example in examples %}
Example {{ loop.index }}:
Instruction: {{ example }}

{% endfor %}
Now, generate a new instruction in a similar style:
'''.rstrip()

text_gen = TextGeneration(
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    template=CUSTOM_TEMPLATE,
    columns="examples",
)

text_gen.load()

result = next(
    text_gen.process(
        [
            {
                "examples": ["This is an example", "Another relevant example"],
                "system_prompt": "You are an AI assisstant specialised in cybersecurity and computing in general, you make your point clear without any explanations."
            }
        ]
    )
)
# result
# [
#     {
#         'examples': ['This is an example', 'Another relevant example'],
#         'system_prompt': 'You are an AI assisstant specialised in cybersecurity and computing in general, you make your point clear without any explanations.',
#         'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
#         'generation': 'Disable the firewall on the router',
#     }
# ]
```




