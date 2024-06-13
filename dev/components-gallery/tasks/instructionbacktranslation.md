---
hide:
  - navigation
---
# InstructionBacktranslation

Self-Alignment with Instruction Backtranslation.







### Attributes

- **_template**: the Jinja2 template to use for the Instruction Backtranslation task.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[generation]
		end
		subgraph New columns
			OCOL0[score]
			OCOL1[reason]
			OCOL2[model_name]
		end
	end

	subgraph InstructionBacktranslation
		StepInput[Input Columns: instruction, generation]
		StepOutput[Output Columns: score, reason, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **instruction** (`str`): The reference instruction to evaluate the text output.

- **generation** (`str`): The text output to evaluate for the given instruction.




#### Outputs


- **score** (`str`): The score for the generation based on the given instruction.

- **reason** (`str`): The reason for the provided score.

- **model_name** (`str`): The model name used to score the generation.







### References

- [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259)


