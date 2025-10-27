---
hide:
  - navigation
---
# ArgillaLabeller

Annotate Argilla records based on input fields, example records and question settings.



This task is designed to facilitate the annotation of Argilla records by leveraging a pre-trained LLM.
    It uses a system prompt that guides the LLM to understand the input fields, the question type,
    and the question settings. The task then formats the input data and generates a response based on the question.
    The response is validated against the question's value model, and the final suggestion is prepared for annotation.





### Attributes

- **_template**: a Jinja2 template used to format the input for the LLM.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[record]
			ICOL1[fields]
			ICOL2[question]
			ICOL3[example_records]
			ICOL4[guidelines]
		end
		subgraph New columns
			OCOL0[suggestion]
		end
	end

	subgraph ArgillaLabeller
		StepInput[Input Columns: record, fields, question, example_records, guidelines]
		StepOutput[Output Columns: suggestion]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	ICOL3 --> StepInput
	ICOL4 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **record** (`argilla.Record`): The record to be annotated.

- **fields** (`Optional[List[Dict[str, Any]]]`): The list of field settings for the input fields.

- **question** (`Optional[Dict[str, Any]]`): The question settings for the question to be answered.

- **example_records** (`Optional[List[Dict[str, Any]]]`): The few shot example records with responses to be used to answer the question.

- **guidelines** (`Optional[str]`): The guidelines for the annotation task.




#### Outputs


- **suggestion** (`Dict[str, Any]`): The final suggestion for annotation.





### Examples


#### Annotate a record with the same dataset and question
```python
import argilla as rg
from argilla import Suggestion
from distilabel.steps.tasks import ArgillaLabeller
from distilabel.models import InferenceEndpointsLLM

# Get information from Argilla dataset definition
dataset = rg.Dataset("my_dataset")
pending_records_filter = rg.Filter(("status", "==", "pending"))
completed_records_filter = rg.Filter(("status", "==", "completed"))
pending_records = list(
    dataset.records(
        query=rg.Query(filter=pending_records_filter),
        limit=5,
    )
)
example_records = list(
    dataset.records(
        query=rg.Query(filter=completed_records_filter),
        limit=5,
    )
)
field = dataset.settings.fields["text"]
question = dataset.settings.questions["label"]

# Initialize the labeller with the model and fields
labeller = ArgillaLabeller(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    fields=[field],
    question=question,
    example_records=example_records,
    guidelines=dataset.guidelines
)
labeller.load()

# Process the pending records
result = next(
    labeller.process(
        [
            {
                "record": record
            } for record in pending_records
        ]
    )
)

# Add the suggestions to the records
for record, suggestion in zip(pending_records, result):
    record.suggestions.add(Suggestion(**suggestion["suggestion"]))

# Log the updated records
dataset.records.log(pending_records)
```

#### Annotate a record with alternating datasets and questions
```python
import argilla as rg
from distilabel.steps.tasks import ArgillaLabeller
from distilabel.models import InferenceEndpointsLLM

# Get information from Argilla dataset definition
dataset = rg.Dataset("my_dataset")
field = dataset.settings.fields["text"]
question = dataset.settings.questions["label"]
question2 = dataset.settings.questions["label2"]

# Initialize the labeller with the model and fields
labeller = ArgillaLabeller(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    )
)
labeller.load()

# Process the record
record = next(dataset.records())
result = next(
    labeller.process(
        [
            {
                "record": record,
                "fields": [field],
                "question": question,
            },
            {
                "record": record,
                "fields": [field],
                "question": question2,
            }
        ]
    )
)

# Add the suggestions to the record
for suggestion in result:
    record.suggestions.add(rg.Suggestion(**suggestion["suggestion"]))

# Log the updated record
dataset.records.log([record])
```

#### Overwrite default prompts and instructions
```python
import argilla as rg
from distilabel.steps.tasks import ArgillaLabeller
from distilabel.models import InferenceEndpointsLLM

# Overwrite default prompts and instructions
labeller = ArgillaLabeller(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    ),
    system_prompt="You are an expert annotator and labelling assistant that understands complex domains and natural language processing.",
    question_to_label_instruction={
        "label_selection": "Select the appropriate label from the list of provided labels.",
        "multi_label_selection": "Select none, one or multiple labels from the list of provided labels.",
        "text": "Provide a text response to the question.",
        "rating": "Provide a rating for the question.",
    },
)
labeller.load()
```




### References

- [Argilla: Argilla is a collaboration tool for AI engineers and domain experts to build high-quality datasets](https://github.com/argilla-io/argilla/)


