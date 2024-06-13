---
hide:
  - navigation
---
# EvolComplexity

Evolve instructions to make them more complex using an `LLM`.



`EvolComplexity` is a task that evolves instructions to make them more complex,
    and it is based in the EvolInstruct task, using slight different prompts, but the
    exact same evolutionary approach.





### Attributes

- **num_instructions**: The number of instructions to be generated.

- **generate_answers**: Whether to generate answers for the instructions or not. Defaults  to `False`.

- **mutation_templates**: The mutation templates to be used for the generation of the  instructions.

- **min_length**: Defines the length (in bytes) that the generated instruction needs to  be higher than, to be considered valid. Defaults to `512`.

- **max_length**: Defines the length (in bytes) that the generated instruction needs to  be lower than, to be considered valid. Defaults to `1024`.

- **seed**: The seed to be set for `numpy` in order to randomly pick a mutation method.  Defaults to `42`.




### Runtime Parameters

- **min_length**: Defines the length (in bytes) that the generated instruction needs to be higher than, to be considered valid.

- **max_length**: Defines the length (in bytes) that the generated instruction needs to be lower than, to be considered valid.

- **seed**: The number of evolutions to be run.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
		end
		subgraph New columns
			OCOL0[evolved_instruction]
			OCOL1[answer]
			OCOL2[model_name]
		end
	end

	subgraph EvolComplexity
		StepInput[Input Columns: instruction]
		StepOutput[Output Columns: evolved_instruction, answer, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The instruction to evolve.




#### Outputs


- **evolved_instruction** (`str`): The evolved instruction.

- **answer** (`str`, optional): The answer to the instruction if `generate_answers=True`.

- **model_name** (`str`): The name of the LLM used to evolve the instructions.





### Examples


#### Evolve an instruction using an LLM
```python
from distilabel.steps.tasks import EvolComplexity
from distilabel.llms.huggingface import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
evol_complexity = EvolComplexity(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    num_evolutions=2,
)

evol_complexity.load()

result = next(evol_complexity.process([{"instruction": "common instruction"}]))
# result
# [{'instruction': 'common instruction', 'evolved_instruction': 'evolved instruction', 'model_name': 'model_name'}]
```




### References

- [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685)

- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)


