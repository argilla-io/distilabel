---
hide:
  - navigation
---
# EvolComplexityGenerator

Generate evolved instructions with increased complexity using an `LLM`.



`EvolComplexityGenerator` is a generation task that evolves instructions to make
    them more complex, and it is based in the EvolInstruct task, but using slight different
    prompts, but the exact same evolutionary approach.





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
		subgraph New columns
			OCOL0[evolved_instruction]
			OCOL1[evolved_instructions]
			OCOL2[model_name]
			OCOL3[answer]
			OCOL4[answers]
		end
	end

	subgraph EvolComplexityGenerator
		StepOutput[Output Columns: evolved_instruction, evolved_instructions, model_name, answer, answers]
	end

	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepOutput --> OCOL3
	StepOutput --> OCOL4

```




#### Outputs


- **evolved_instruction** (`str`): The evolved instruction if `store_evolutions=False`.

- **evolved_instructions** (`List[str]`): The evolved instructions if `store_evolutions=True`.

- **model_name** (`str`): The name of the LLM used to evolve the instructions.

- **answer** (`str`): The answer to the evolved instruction if `generate_answers=True`  and `store_evolutions=False`.

- **answers** (`List[str]`): The answers to the evolved instructions if `generate_answers=True`  and `store_evolutions=True`.





### Examples


#### Generate evolved instructions without initial instructions
```python
from distilabel.steps.tasks import EvolComplexityGenerator
from distilabel.llms.huggingface import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
evol_complexity_generator = EvolComplexityGenerator(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    num_instructions=2,
)

evol_complexity_generator.load()

result = next(scorer.process())
# result
# [{'instruction': 'generated instruction', 'model_name': 'test'}]
```




### References

- [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685)

- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)


