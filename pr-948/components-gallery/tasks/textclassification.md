---
hide:
  - navigation
---
# TextClassification

Classifies text into one or more categories or labels.



ADD EXAMPLES, THIS IS HIGHLY CUSTOMIZABLE.
    It uses structured generation as per the reference paper, it can help to generate more
    concise labels. See section 4.1 in the reference.





### Attributes

- **system_prompt**: A prompt to display to the user before the task starts. Contains a default  message to make the model behave like a classifier specialist.

- **n**: Number of labels to generate If only 1 is required, corresponds to a label  classification problem, if >1 it will intend return the "n" labels most representative  for the text. Defaults to 1.

- **context**: Context to use when generating the labels. By default contains a generic message,  but can be used to customize the context for the task.

- **examples**: List of examples to help the model understand the task, few shots.

- **available_labels**: List of available labels to choose from when classifying the text, or  a dictionary with the labels and their descriptions.

- **default_label**: Default label to use when the text is ambiguous or lacks sufficient information for  classification. Can be a list in case of multiple labels (n>1).





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[text]
		end
		subgraph New columns
			OCOL0[labels]
			OCOL1[model_name]
		end
	end

	subgraph TextClassification
		StepInput[Input Columns: text]
		StepOutput[Output Columns: labels, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **text** (`str`): The reference text we want to obtain labels for.




#### Outputs


- **labels** (`Union[str, List[str]]`): The label or list of labels for the text.

- **model_name** (`str`): The name of the model used to generate the label/s.







### References

- [Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models](https://arxiv.org/abs/2408.02442)


