---
hide:
  - navigation
---
# EvolQuality

Evolve the quality of the responses using an `LLM`.



`EvolQuality` task is used to evolve the quality of the responses given a prompt,
    by generating a new response with a language model. This step implements the evolution
    quality task from the paper 'What Makes Good Data for Alignment? A Comprehensive Study of
    Automatic Data Selection in Instruction Tuning'.





### Attributes

- **num_evolutions**: The number of evolutions to be performed on the responses.

- **store_evolutions**: Whether to store all the evolved responses or just the last one.  Defaults to `False`.

- **include_original_response**: Whether to include the original response within the evolved  responses. Defaults to `False`.

- **mutation_templates**: The mutation templates to be used to evolve the responses.

- **seed**: The seed to be set for `numpy` in order to randomly pick a mutation method.  Defaults to `42`.




### Runtime Parameters

- **seed**: The seed to be set for `numpy` in order to randomly pick a mutation method.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[response]
		end
		subgraph New columns
			OCOL0[evolved_response]
			OCOL1[evolved_responses]
			OCOL2[model_name]
		end
	end

	subgraph EvolQuality
		StepInput[Input Columns: instruction, response]
		StepOutput[Output Columns: evolved_response, evolved_responses, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The instruction that was used to generate the `responses`.

- **response** (`str`): The responses to be rewritten.




#### Outputs


- **evolved_response** (`str`): The evolved response if `store_evolutions=False`.

- **evolved_responses** (`List[str]`): The evolved responses if `store_evolutions=True`.

- **model_name** (`str`): The name of the LLM used to evolve the responses.





### Examples


#### Evolve the quality of the responses given a prompt
```python
from distilabel.steps.tasks import EvolQuality
from distilabel.llms.huggingface import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
evol_quality = EvolQuality(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    num_evolutions=2,
)

evol_quality.load()

result = next(
    evol_quality.process(
        [
            {"instruction": "common instruction", "response": "a response"},
        ]
    )
)
# result
# [
#     {
#         'instruction': 'common instruction',
#         'response': 'a response',
#         'evolved_response': 'evolved response',
#         'model_name': '"mistralai/Mistral-7B-Instruct-v0.2"'
#     }
# ]
```




### References

- [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685)


