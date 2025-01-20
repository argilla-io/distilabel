---
hide:
  - navigation
---
# EvolInstruct

Evolve instructions using an `LLM`.



WizardLM: Empowering Large Language Models to Follow Complex Instructions





### Attributes

- **num_evolutions**: The number of evolutions to be performed.

- **store_evolutions**: Whether to store all the evolutions or just the last one. Defaults  to `False`.

- **generate_answers**: Whether to generate answers for the evolved instructions. Defaults  to `False`.

- **include_original_instruction**: Whether to include the original instruction in the  `evolved_instructions` output column. Defaults to `False`.

- **mutation_templates**: The mutation templates to be used for evolving the instructions.  Defaults to the ones provided in the `utils.py` file.

- **seed**: The seed to be set for `numpy` in order to randomly pick a mutation method.  Defaults to `42`.




### Runtime Parameters

- **seed**: The seed to be set for `numpy` in order to randomly pick a mutation method.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
		end
		subgraph New columns
			OCOL0[evolved_instruction]
			OCOL1[evolved_instructions]
			OCOL2[model_name]
			OCOL3[answer]
			OCOL4[answers]
		end
	end

	subgraph EvolInstruct
		StepInput[Input Columns: instruction]
		StepOutput[Output Columns: evolved_instruction, evolved_instructions, model_name, answer, answers]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepOutput --> OCOL3
	StepOutput --> OCOL4
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The instruction to evolve.




#### Outputs


- **evolved_instruction** (`str`): The evolved instruction if `store_evolutions=False`.

- **evolved_instructions** (`List[str]`): The evolved instructions if `store_evolutions=True`.

- **model_name** (`str`): The name of the LLM used to evolve the instructions.

- **answer** (`str`): The answer to the evolved instruction if `generate_answers=True`  and `store_evolutions=False`.

- **answers** (`List[str]`): The answers to the evolved instructions if `generate_answers=True`  and `store_evolutions=True`.





### Examples


#### Evolve an instruction using an LLM
```python
from distilabel.steps.tasks import EvolInstruct
from distilabel.models import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
evol_instruct = EvolInstruct(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    num_evolutions=2,
)

evol_instruct.load()

result = next(evol_instruct.process([{"instruction": "common instruction"}]))
# result
# [{'instruction': 'common instruction', 'evolved_instruction': 'evolved instruction', 'model_name': 'model_name'}]
```

#### Keep the iterations of the evolutions
```python
from distilabel.steps.tasks import EvolInstruct
from distilabel.models import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
evol_instruct = EvolInstruct(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    num_evolutions=2,
    store_evolutions=True,
)

evol_instruct.load()

result = next(evol_instruct.process([{"instruction": "common instruction"}]))
# result
# [
#     {
#         'instruction': 'common instruction',
#         'evolved_instructions': ['initial evolution', 'final evolution'],
#         'model_name': 'model_name'
#     }
# ]
```

#### Generate answers for the instructions in a single step
```python
from distilabel.steps.tasks import EvolInstruct
from distilabel.models import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
evol_instruct = EvolInstruct(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    num_evolutions=2,
    generate_answers=True,
)

evol_instruct.load()

result = next(evol_instruct.process([{"instruction": "common instruction"}]))
# result
# [
#     {
#         'instruction': 'common instruction',
#         'evolved_instruction': 'evolved instruction',
#         'answer': 'answer to the instruction',
#         'model_name': 'model_name'
#     }
# ]
```




### References

- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)

- [GitHub: h2oai/h2o-wizardlm](https://github.com/h2oai/h2o-wizardlm)


