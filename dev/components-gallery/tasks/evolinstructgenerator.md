---
hide:
  - navigation
---
# EvolInstructGenerator

Generate evolved instructions using an `LLM`.



WizardLM: Empowering Large Language Models to Follow Complex Instructions





### Attributes

- **num_instructions**: The number of instructions to be generated.

- **generate_answers**: Whether to generate answers for the instructions or not. Defaults  to `False`.

- **mutation_templates**: The mutation templates to be used for the generation of the  instructions.

- **min_length**: Defines the length (in bytes) that the generated instruction needs to  be higher than, to be considered valid. Defaults to `512`.

- **max_length**: Defines the length (in bytes) that the generated instruction needs to  be lower than, to be considered valid. Defaults to `1024`.

- **seed**: The seed to be set for `numpy` in order to randomly pick a mutation method.  Defaults to `42`.




### Runtime Parameters

- **min_length**: Defines the length (in bytes) that the generated instruction needs  to be higher than, to be considered valid.

- **max_length**: Defines the length (in bytes) that the generated instruction needs  to be lower than, to be considered valid.

- **seed**: The seed to be set for `numpy` in order to randomly pick a mutation method.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph New columns
			OCOL0[instruction]
			OCOL1[answer]
			OCOL2[instructions]
			OCOL3[model_name]
		end
	end

	subgraph EvolInstructGenerator
		StepOutput[Output Columns: instruction, answer, instructions, model_name]
	end

	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepOutput --> OCOL3

```




#### Outputs


- **instruction** (`str`): The generated instruction if `generate_answers=False`.

- **answer** (`str`): The generated answer if `generate_answers=True`.

- **instructions** (`List[str]`): The generated instructions if `generate_answers=True`.

- **model_name** (`str`): The name of the LLM used to generate and evolve the instructions.





### Examples


#### Generate evolved instructions without initial instructions
```python
from distilabel.steps.tasks import EvolInstructGenerator
from distilabel.models import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
evol_instruct_generator = EvolInstructGenerator(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    num_instructions=2,
)

evol_instruct_generator.load()

result = next(scorer.process())
# result
# [{'instruction': 'generated instruction', 'model_name': 'test'}]
```




### References

- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)

- [GitHub: h2oai/h2o-wizardlm](https://github.com/h2oai/h2o-wizardlm)


