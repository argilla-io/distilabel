---
hide:
  - navigation
---
# TextGenerationWithCotReflection

Text generation with an `LLM` using Chain of Thought (CoT) reflection.



`TextGenerationWithCotReflection` is a `Task` that allows generating a response for
    a given instruction using a Chain of Thought (CoT) approach with reflection. The `LLM`
    will first think through the problem step by step, reflect on the thinking process, make
    any necessary adjustments based on the reflection, and provide a final, concise answer.
    This method usually helps in generating more accurate and thoughtful responses at the
    cost of generating more tokens and being slower.





### Attributes

- **system_prompt**: The system prompt to use in the generation and that will be appended  to the CoT Reflection system prompt. If not provided, then it will check if  the input row has a column named `system_prompt` and use it. If not, then no  system prompt will be used. Defaults to `None`.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[system_prompt]
		end
		subgraph New columns
			OCOL0[thinking]
		end
	end

	subgraph TextGenerationWithCotReflection
		StepInput[Input Columns: instruction, system_prompt]
		StepOutput[Output Columns: thinking]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The instruction to generate the response.

- **system_prompt** (`str`, optional): The system prompt to use in the generation and  that will be appended to the CoT Reflection system prompt. Defaults to `None`.




#### Outputs


- **thinking** (`str`): The step-by-step reasoning process.





### Examples


#### Generate text from an instruction
```python
from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import TextGenerationWithCotReflection

task = TextGenerationWithCotReflection(
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        generation_kwargs={"temperature": 0.8, "max_new_tokens": 2048},
    ),
    use_cache=False,
)

task.load()


result = next(
    task.process_applying_mappings(
        [
            {
                "instruction": "If all cats have whiskers, and Fluffy is a cat, but Fluffy doesn't have whiskers, what can we conclude about this situation?"
            }
        ]
    )
)
# {
#     "instruction": "If all cats have whiskers, and Fluffy is a cat, but Fluffy doesn't have whiskers, what can we conclude about this situation?",
#     "thinking": "Let's break down the information provided: 
- All cats have whiskers.
- Fluffy is a cat.
- Fluffy doesn't have whiskers...",
#     "output": 'We can conclude that either the general rule "all cats have whiskers" is incorrect, ...',
# }
```




