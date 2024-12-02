---
hide:
  - navigation
---
# MathShepherdGenerator

Math Shepherd solution generator.



This task is in charge of generating completions for a given instruction, in the format expected
    by the Math Shepherd Completer task. The attributes make the task flexible to be used with different
    types of dataset and LLMs, but we provide examples for the GSM8K and MATH datasets as presented
    in the original paper. Before modifying them, review the current defaults to ensure the completions
    are generated correctly. This task can be used to generate the golden solutions for a given problem if
    not provided, as well as possible solutions to be then labeled by the Math Shepherd Completer.
    Only one of `solutions` or `golden_solution` will be generated, depending on the value of M.





### Attributes

- **system_prompt**: The system prompt to be used in the completions. The default one has been  checked and generates good completions using Llama 3.1 with 8B and 70B,  but it can be modified to adapt it to the model and dataset selected.  Take into account that the system prompt includes 2 variables in the Jinja2 template,  {{extra_rules}} and {{few_shot}}. These variables are used to include extra rules, for example  to steer the model towards a specific type of responses, and few shots to add examples.  They can be modified to adapt the system prompt to the dataset and model used without needing  to change the full system prompt.

- **extra_rules**: This field can be used to insert extra rules relevant to the type of dataset.  For example, in the original paper they used GSM8K and MATH datasets, and this field  can be used to insert the rules for the GSM8K dataset.

- **few_shots**: Few shots to help the model generating the completions, write them in the  format of the type of solutions wanted for your dataset.

- **M**: Number of completions to generate for each step. By default is set to 1, which will  generate the "golden_solution". In this case select a stronger model, as it will be used  as the source of true during labelling. If M is set to a number greater than 1, the task  will generate a list of completions to be labeled by the Math Shepherd Completer task.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
		end
		subgraph New columns
			OCOL0[golden_solution]
			OCOL1[solutions]
			OCOL2[model_name]
		end
	end

	subgraph MathShepherdGenerator
		StepInput[Input Columns: instruction]
		StepOutput[Output Columns: golden_solution, solutions, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The task or instruction.




#### Outputs


- **golden_solution** (`str`): The step by step solution to the instruction.  It will be generated if M is equal to 1.

- **solutions** (`List[List[str]]`): A list of possible solutions to the instruction.  It will be generated if M is greater than 1.

- **model_name** (`str`): The name of the model used to generate the revision.





### Examples


#### Generate the solution for a given instruction (prefer a stronger model here)
```python
from distilabel.steps.tasks import MathShepherdGenerator
from distilabel.models import InferenceEndpointsLLM

llm=InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    generation_kwargs={
        "temperature": 0.6,
        "max_new_tokens": 1024,
    },
)
task = MathShepherdGenerator(
    name="golden_solution_generator",
    llm=llm,
)

task.load()

result = next(
    task.process(
        [
            {
                "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            },
        ]
    )
)
# [[{'instruction': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
# 'golden_solution': '["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.", "The answer is: 18"]'}]]
```

#### Generate M completions for a given instruction (using structured output generation)
```python
from distilabel.steps.tasks import MathShepherdGenerator
from distilabel.models import InferenceEndpointsLLM

llm=InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    generation_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 2048,
    },
)
task = MathShepherdGenerator(
    name="solution_generator",
    llm=llm,
    M=2,
    use_default_structured_output=True,
)

task.load()

result = next(
    task.process(
        [
            {
                "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            },
        ]
    )
)
# [[{'instruction': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
# 'solutions': [["Step 1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. -", "Step 2: She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.", "The answer is: 18"], ["Step 1: Janets ducks lay 16 eggs per day, and she uses 3 + 4 = <<3+4=7>>7 for eating and baking. +", "Step 2: So she sells 16 - 7 = <<16-7=9>>9 duck eggs every day. +", "Step 3: Those 9 eggs are worth 9 * $2 = $<<9*2=18>>18.", "The answer is: 18"]]}]]
```




### References

- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)


