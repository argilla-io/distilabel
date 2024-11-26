---
hide:
  - navigation
---
# MathShepherdCompleter

Math Shepherd Completer and auto-labeller task.



This task is in charge of, given a list of solutions to an instruction, and a golden solution,
    as reference, generate completions for the solutions, and label them according to the golden
    solution using the hard estimation method from figure 2 in the reference paper, Eq. 3.
    The attributes make the task flexible to be used with different types of dataset and LLMs, and
    allow making use of different fields to modify the system and user prompts for it. Before modifying
    them, review the current defaults to ensure the completions are generated correctly.





### Attributes

- **system_prompt**: The system prompt to be used in the completions. The default one has been  checked and generates good completions using Llama 3.1 with 8B and 70B,  but it can be modified to adapt it to the model and dataset selected.

- **extra_rules**: This field can be used to insert extra rules relevant to the type of dataset.  For example, in the original paper they used GSM8K and MATH datasets, and this field  can be used to insert the rules for the GSM8K dataset.

- **few_shots**: Few shots to help the model generating the completions, write them in the  format of the type of solutions wanted for your dataset.

- **N**: Number of completions to generate for each step, correspond to N in the paper.  They used 8 in the paper, but it can be adjusted.

- **tags**: List of tags to be used in the completions, the default ones are ["+", "-"] as in the  paper, where the first is used as a positive label, and the second as a negative one.  This can be updated, but it MUST be a list with 2 elements, where the first is the  positive one, and the second the negative one.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[solutions]
			ICOL2[golden_solution]
		end
		subgraph New columns
			OCOL0[solutions]
			OCOL1[model_name]
		end
	end

	subgraph MathShepherdCompleter
		StepInput[Input Columns: instruction, solutions, golden_solution]
		StepOutput[Output Columns: solutions, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The task or instruction.

- **solutions** (`str`): JSON formatted list of solutions to the task.

- **golden_solution** (`str`): The reference solution to the task, will be used  to annotate the candidate solutions.




#### Outputs


- **solutions** (`str`): The same columns that were used as input, the "solutions" is modified.

- **model_name** (`str`): The name of the model used to generate the revision.





### Examples


#### Annotate your steps with the Math Shepherd Completer
```python
from distilabel.steps.tasks import MathShepherdCompleter
from distilabel.models import InferenceEndpointsLLM

llm=InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    generation_kwargs={
        "temperature": 0.6,
        "max_new_tokens": 1024,
    },
)
task = MathShepherdCompleter(
    llm=llm,
    N=3
)

task.load()

result = next(
    task.process(
        [
            {
                "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "golden_solution": ["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.", "The answer is: 18"],
                "solutions": [
                    ["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.", "The answer is: 18"],
                    ['Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking.', 'Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day.', 'Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18.', 'The answer is: 18'],
                ]
            },
        ]
    )
)
# [[{'instruction': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
# 'golden_solution': ["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.", "The answer is: 18"],
# 'solutions': [["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. -", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.", "The answer is: 18"], ["Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking. +", "Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day. +", "Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18.", "The answer is: 18"]]}]]
```




### References

- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)


