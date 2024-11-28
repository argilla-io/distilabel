---
hide:
  - navigation
---
# FormatChatGenerationDPO

Format the output of a combination of a `ChatGeneration` + a preference task for Direct Preference Optimization (DPO).



`FormatChatGenerationDPO` is a `Step` that formats the output of the combination of a `ChatGeneration`
    task with a preference `Task` i.e. a task generating `ratings` such as `UltraFeedback` following the standard
    formatting from frameworks such as `axolotl` or `alignment-handbook`., so that those are used to rank the
    existing generations and provide the `chosen` and `rejected` generations based on the `ratings`.



### Note
The `messages` column should contain at least one message from the user, the `generations`
column should contain at least two generations, the `ratings` column should contain the same
number of ratings as generations.






### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[messages]
			ICOL1[generations]
			ICOL2[generation_models]
			ICOL3[ratings]
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

	subgraph FormatChatGenerationDPO
		StepInput[Input Columns: messages, generations, generation_models, ratings]
		StepOutput[Output Columns: prompt, prompt_id, chosen, chosen_model, chosen_rating, rejected, rejected_model, rejected_rating]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	ICOL3 --> StepInput
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


- **messages** (`List[Dict[str, str]]`): The conversation messages.

- **generations** (`List[str]`): The generations produced by the `LLM`.

- **generation_models** (`List[str]`, optional): The model names used to generate the `generations`,  only available if the `model_name` from the `ChatGeneration` task/s is combined into a single  column named this way, otherwise, it will be ignored.

- **ratings** (`List[float]`): The ratings for each of the `generations`, produced by a preference  task such as `UltraFeedback`.




#### Outputs


- **prompt** (`str`): The user message used to generate the `generations` with the `LLM`.

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
from distilabel.steps import FormatChatGenerationDPO

format_dpo = FormatChatGenerationDPO()
format_dpo.load()

# NOTE: "generation_models" can be added optionally.
result = next(
    format_dpo.process(
        [
            {
                "messages": [{"role": "user", "content": "What's 2+2?"}],
                "generations": ["4", "5", "6"],
                "ratings": [1, 0, -1],
            }
        ]
    )
)
# >>> result
# [
#     {
#         'messages': [{'role': 'user', 'content': "What's 2+2?"}],
#         'generations': ['4', '5', '6'],
#         'ratings': [1, 0, -1],
#         'prompt': "What's 2+2?",
#         'prompt_id': '7762ecf17ad41479767061a8f4a7bfa3b63d371672af5180872f9b82b4cd4e29',
#         'chosen': [{'role': 'user', 'content': "What's 2+2?"}, {'role': 'assistant', 'content': '4'}],
#         'chosen_rating': 1,
#         'rejected': [{'role': 'user', 'content': "What's 2+2?"}, {'role': 'assistant', 'content': '6'}],
#         'rejected_rating': -1
#     }
# ]
```




