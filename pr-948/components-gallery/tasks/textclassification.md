---
hide:
  - navigation
---
# TextClassification

Classifies text into one or more categories or labels.



This task can be used for text classification problems, where the goal is to assign
    one or multiple labels to a given text.
    It uses structured generation as per the reference paper by default,
    it can help to generate more concise labels. See section 4.1 in the reference.





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





### Examples


#### Assigning a sentiment to a text
```python
from distilabel.steps.tasks import TextClassification
from distilabel.llms.huggingface import InferenceEndpointsLLM

llm = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
)

text_classification = TextClassification(
    llm=llm,
    context="You are an AI system specialized in assigning sentiment to movies.",
    available_labels=["positive", "negative"],
)

text_classification.load()

result = next(
    text_classification.process(
        [{"text": "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."}]
    )
)
# result
# [{'text': 'This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three.',
# 'labels': 'positive',
# 'distilabel_metadata': {'raw_output_text_classification_0': '{\n    "labels": "positive"\n}',
# 'raw_input_text_classification_0': [{'role': 'system',
#     'content': 'You are an AI system specialized in generating labels to classify pieces of text. Your sole purpose is to analyze the given text and provide appropriate classification labels.'},
#     {'role': 'user',
#     'content': '# Instruction\nPlease classify the user query by assigning the most appropriate labels.\nDo not explain your reasoning or provide any additional commentary.\nIf the text is ambiguous or lacks sufficient information for classification, respond with "Unclassified".\nProvide the label that best describes the text.\nYou are an AI system specialized in assigning sentiment to movie the user queries.\n## Labeling the user input\nUse the available labels to classify the user query. Analyze the context of each label specifically:\navailable_labels = [\n    "positive",  # The text shows positive sentiment\n    "negative",  # The text shows negative sentiment\n]\n\n\n## User Query\n```\nThis was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three.\n```\n\n## Output Format\nNow, please give me the labels in JSON format, do not include any other text in your response:\n```\n{\n    "labels": "label"\n}\n```'}]},
# 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
```

#### Assigning predefined labels with specified descriptions
```python
from distilabel.steps.tasks import TextClassification

text_classification = TextClassification(
    llm=llm,
    n=1,
    context="Determine the intent of the text.",
    available_labels={
        "complaint": "A statement expressing dissatisfaction or annoyance about a product, service, or experience. It's a negative expression of discontent, often with the intention of seeking a resolution or compensation.",
        "inquiry": "A question or request for information about a product, service, or situation. It's a neutral or curious expression seeking clarification or details.",
        "feedback": "A statement providing evaluation, opinion, or suggestion about a product, service, or experience. It can be positive, negative, or neutral, and is often intended to help improve or inform.",
        "praise": "A statement expressing admiration, approval, or appreciation for a product, service, or experience. It's a positive expression of satisfaction or delight, often with the intention of encouraging or recommending."
    },
    query_title="Customer Query",
)

text_classification.load()

result = next(
    text_classification.process(
        [{"text": "Can you tell me more about your return policy?"}]
    )
)
# result
# [{'text': 'Can you tell me more about your return policy?',
# 'labels': 'inquiry',
# 'distilabel_metadata': {'raw_output_text_classification_0': '{\n    "labels": "inquiry"\n}',
# 'raw_input_text_classification_0': [{'role': 'system',
#     'content': 'You are an AI system specialized in generating labels to classify pieces of text. Your sole purpose is to analyze the given text and provide appropriate classification labels.'},
#     {'role': 'user',
#     'content': '# Instruction\nPlease classify the customer query by assigning the most appropriate labels.\nDo not explain your reasoning or provide any additional commentary.\nIf the text is ambiguous or lacks sufficient information for classification, respond with "Unclassified".\nProvide the label that best describes the text.\nDetermine the intent of the text.\n## Labeling the user input\nUse the available labels to classify the user query. Analyze the context of each label specifically:\navailable_labels = [\n    "complaint",  # A statement expressing dissatisfaction or annoyance about a product, service, or experience. It\'s a negative expression of discontent, often with the intention of seeking a resolution or compensation.\n    "inquiry",  # A question or request for information about a product, service, or situation. It\'s a neutral or curious expression seeking clarification or details.\n    "feedback",  # A statement providing evaluation, opinion, or suggestion about a product, service, or experience. It can be positive, negative, or neutral, and is often intended to help improve or inform.\n    "praise",  # A statement expressing admiration, approval, or appreciation for a product, service, or experience. It\'s a positive expression of satisfaction or delight, often with the intention of encouraging or recommending.\n]\n\n\n## Customer Query\n```\nCan you tell me more about your return policy?\n```\n\n## Output Format\nNow, please give me the labels in JSON format, do not include any other text in your response:\n```\n{\n    "labels": "label"\n}\n```'}]},
# 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
```

#### Free multi label classification without predefined labels
```python
from distilabel.steps.tasks import TextClassification

text_classification = TextClassification(
    llm=llm,
    n=3,
    context=(
        "Describe the main themes, topics, or categories that could describe the "
        "following type of persona."
    ),
    query_title="Example of Persona",
)

text_classification.load()

result = next(
    text_classification.process(
        [{"text": "A historian or curator of Mexican-American history and culture focused on the cultural, social, and historical impact of the Mexican presence in the United States."}]
    )
)
# result
# [{'text': 'A historian or curator of Mexican-American history and culture focused on the cultural, social, and historical impact of the Mexican presence in the United States.',
# 'labels': ['Historical Researcher',
# 'Cultural Specialist',
# 'Ethnic Studies Expert'],
# 'distilabel_metadata': {'raw_output_text_classification_0': '{\n    "labels": ["Historical Researcher", "Cultural Specialist", "Ethnic Studies Expert"]\n}',
# 'raw_input_text_classification_0': [{'role': 'system',
#     'content': 'You are an AI system specialized in generating labels to classify pieces of text. Your sole purpose is to analyze the given text and provide appropriate classification labels.'},
#     {'role': 'user',
#     'content': '# Instruction\nPlease classify the example of persona by assigning the most appropriate labels.\nDo not explain your reasoning or provide any additional commentary.\nIf the text is ambiguous or lacks sufficient information for classification, respond with "Unclassified".\nProvide a list of 3 labels that best describe the text.\nDescribe the main themes, topics, or categories that could describe the following type of persona.\nUse clear, widely understood terms for labels.Avoid overly specific or obscure labels unless the text demands it.\n\n\n## Example of Persona\n```\nA historian or curator of Mexican-American history and culture focused on the cultural, social, and historical impact of the Mexican presence in the United States.\n```\n\n## Output Format\nNow, please give me the labels in JSON format, do not include any other text in your response:\n```\n{\n    "labels": ["label_0", "label_1", "label_2"]\n}\n```'}]},
# 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
```




### References

- [Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models](https://arxiv.org/abs/2408.02442)


