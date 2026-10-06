---
hide:
  - navigation
---
# FormatTextGenerationDPO

Format the output of your LLMs for Direct Preference Optimization (DPO).



`FormatTextGenerationDPO` is a `Step` that formats the output of the combination of a `TextGeneration`
    task with a preference `Task` i.e. a task generating `ratings`, so that those are used to rank the
    existing generations and provide the `chosen` and `rejected` generations based on the `ratings`.
    Use this step to transform the output of a combination of a `TextGeneration` + a preference task such as
    `UltraFeedback` following the standard formatting from frameworks such as `axolotl` or `alignment-handbook`.



### Note
The `generations` column should contain at least two generations, the `ratings` column should
contain the same number of ratings as generations.






### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[system_prompt]
			ICOL1[instruction]
			ICOL2[generations]
			ICOL3[generation_models]
			ICOL4[ratings]
		end
		subgraph New columns
			OCOL0[prompt]
			OCOL1[prompt_id]
			OCOL2[chosen]
			OCOL3[chosen_model]
			OCOL4[chosen_rating]
			OCOL5[rejected]
			OCOL6[rejected_model]
			OCOL7[rejected_rating]
		end
	end

	subgraph FormatTextGenerationDPO
		StepInput[Input Columns: system_prompt, instruction, generations, generation_models, ratings]
		StepOutput[Output Columns: prompt, prompt_id, chosen, chosen_model, chosen_rating, rejected, rejected_model, rejected_rating]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	ICOL3 --> StepInput
	ICOL4 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepOutput --> OCOL3
	StepOutput --> OCOL4
	StepOutput --> OCOL5
	StepOutput --> OCOL6
	StepOutput --> OCOL7
	StepInput --> StepOutput

```


#### Inputs


- **system_prompt** (`str`, optional): The system prompt used within the `LLM` to generate the  `generations`, if available.

- **instruction** (`str`): The instruction used to generate the `generations` with the `LLM`.

- **generations** (`List[str]`): The generations produced by the `LLM`.

- **generation_models** (`List[str]`, optional): The model names used to generate the `generations`,  only available if the `model_name` from the `TextGeneration` task/s is combined into a single  column named this way, otherwise, it will be ignored.

- **ratings** (`List[float]`): The ratings for each of the `generations`, produced by a preference  task such as `UltraFeedback`.




#### Outputs


- **prompt** (`str`): The instruction used to generate the `generations` with the `LLM`.

- **prompt_id** (`str`): The `SHA256` hash of the `prompt`.

- **chosen** (`List[Dict[str, str]]`): The `chosen` generation based on the `ratings`.

- **chosen_model** (`str`, optional): The model name used to generate the `chosen` generation,  if the `generation_models` are available.

- **chosen_rating** (`float`): The rating of the `chosen` generation.

- **rejected** (`List[Dict[str, str]]`): The `rejected` generation based on the `ratings`.

- **rejected_model** (`str`, optional): The model name used to generate the `rejected` generation,  if the `generation_models` are available.

- **rejected_rating** (`float`): The rating of the `rejected` generation.





### Examples


#### Format your dataset for DPO fine tuning
```python
from distilabel.steps import FormatTextGenerationDPO

format_dpo = FormatTextGenerationDPO()
format_dpo.load()

# NOTE: Both "system_prompt" and "generation_models" can be added optionally.
result = next(
    format_dpo.process(
        [
            {
                "instruction": "What's 2+2?",
                "generations": ["4", "5", "6"],
                "ratings": [1, 0, -1],
            }
        ]
    )
)
# >>> result
# [
#    {   'instruction': "What's 2+2?",
#        'generations': ['4', '5', '6'],
#        'ratings': [1, 0, -1],
#        'prompt': "What's 2+2?",
#        'prompt_id': '7762ecf17ad41479767061a8f4a7bfa3b63d371672af5180872f9b82b4cd4e29',
#        'chosen': [{'role': 'user', 'content': "What's 2+2?"}, {'role': 'assistant', 'content': '4'}],
#        'chosen_rating': 1,
#        'rejected': [{'role': 'user', 'content': "What's 2+2?"}, {'role': 'assistant', 'content': '6'}],
#        'rejected_rating': -1
#    }
# ]
```




