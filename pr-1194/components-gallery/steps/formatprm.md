---
hide:
  - navigation
---
# FormatPRM

Helper step to transform the data into the format expected by the PRM model.



This step can be used to format the data in one of 2 formats:
    Following the format presented
    in [peiyi9979/Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd?row=0),
    in which case this step creates the columns input and label, where the input is the instruction
    with the solution (and the tag replaced by a token), and the label is the instruction
    with the solution, both separated by a newline.
    Following TRL's format for training, which generates the columns prompt, completions, and labels.
    The labels correspond to the original tags replaced by boolean values, where True represents
    correct steps.





### Attributes

- **format**: The format to use for the PRM model.  "math-shepherd" corresponds to the original paper, while "trl" is a format  prepared to train the model using TRL.

- **step_token**: String that serves as a unique token denoting the position  for predicting the step score.

- **tags**: List of tags that represent the correct and incorrect steps.  This only needs to be informed if it's different than the default in  `MathShepherdCompleter`.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[solutions]
		end
		subgraph New columns
			OCOL0[input]
			OCOL1[label]
			OCOL2[prompt]
			OCOL3[completions]
			OCOL4[labels]
		end
	end

	subgraph FormatPRM
		StepInput[Input Columns: instruction, solutions]
		StepOutput[Output Columns: input, label, prompt, completions, labels]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepOutput --> OCOL3
	StepOutput --> OCOL4
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The task or instruction.

- **solutions** (`list[str]`): List of steps with a solution to the task.




#### Outputs


- **input** (`str`): The instruction with the solutions, where the label tags  are replaced by a token.

- **label** (`str`): The instruction with the solutions.

- **prompt** (`str`): The instruction with the solutions, where the label tags  are replaced by a token.

- **completions** (`List[str]`): The solution represented as a list of steps.

- **labels** (`List[bool]`): The labels, as a list of booleans, where True represents  a good response.





### Examples


#### Shepherd format
```python
from distilabel.steps.tasks import FormatPRM
from distilabel.steps import ExpandColumns

expand_columns = ExpandColumns(columns=["solutions"])
expand_columns.load()

# Define our PRM formatter
formatter = FormatPRM()
formatter.load()

# Expand the solutions column as it comes from the MathShepherdCompleter
result = next(
    expand_columns.process(
        [
            {
                "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "solutions": [["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"]]
            },
        ]
    )
)
result = next(formatter.process(result))
```

#### Prepare your data to train a PRM model with the TRL format
```python
from distilabel.steps.tasks import FormatPRM
from distilabel.steps import ExpandColumns

expand_columns = ExpandColumns(columns=["solutions"])
expand_columns.load()

# Define our PRM formatter
formatter = FormatPRM(format="trl")
formatter.load()

# Expand the solutions column as it comes from the MathShepherdCompleter
result = next(
    expand_columns.process(
        [
            {
                "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "solutions": [["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"]]
            },
        ]
    )
)

result = next(formatter.process(result))
# {
#     "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#     "solutions": [
#         "Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +",
#         "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +",
#         "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"
#     ],
#     "prompt": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#     "completions": [
#         "Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required.",
#         "Step 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber.",
#         "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3"
#     ],
#     "labels": [
#         true,
#         true,
#         true
#     ]
# }
```




### References

- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)

- [peiyi9979/Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd?row=0)


