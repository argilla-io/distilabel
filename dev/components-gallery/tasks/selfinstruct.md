# SelfInstruct


Generate instructions based on a given input using an `LLM`.



`SelfInstruct` is a pre-defined task that, given a number of instructions, a
    certain criteria for query generations, an application description, and an input,
    generates a number of instruction related to the given input and following what
    is stated in the criteria for query generation and the application description.
    It is based in the SelfInstruct framework from the paper "Self-Instruct: Aligning
    Language Models with Self-Generated Instructions".





### Attributes

- **num_instructions**: The number of instructions to be generated. Defaults to 5.

- **criteria_for_query_generation**: The criteria for the query generation. Defaults  to the criteria defined within the paper.

- **application_description**: The description of the AI application that one want  to build with these instructions. Defaults to `AI assistant`.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[input]
		end
		subgraph New columns
			OCOL0[instructions]
			OCOL1[model_name]
		end
	end

	subgraph SelfInstruct
		StepInput[Input Columns: input]
		StepOutput[Output Columns: instructions, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **input** (`str`): The input to generate the instructions. It's also called seed in  the paper.




#### Outputs


- **instructions** (`List[str]`): The generated instructions.

- **model_name** (`str`): The model name used to generate the instructions.







