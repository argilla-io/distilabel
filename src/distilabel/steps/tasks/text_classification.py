# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from textwrap import indent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import orjson
from jinja2 import Template
from pydantic import BaseModel, Field, PositiveInt, PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


TEXT_CLASSIFICATION_TEMPLATE: str = """\
# Instruction
Please classify the user query by assigning the most appropriate labels.
Do not explain your reasoning or provide any additional commentary.
If the text is ambiguous or lacks sufficient information for classification, respond with "{{ default_label }}".
{{ labels_message }}{{ context}}
{{ available_labels }}
{{ examples }}

## User Query
```
{{ text }}
```

## Output Format
Now, please give me the labels in JSON format, do not include any other text in your response:
```
{
    "labels": {{ labels_format }}
}
```
""".rstrip()


class TextClassification(Task):
    """Classifies text into one or more categories or labels.

    ADD EXAMPLES, THIS IS HIGHLY CUSTOMIZABLE.
    It uses structured generation as per the reference paper, it can help to generate more
    concise labels. See section 4.1 in the reference.

    Input columns:
        - text (`str`): The reference text we want to obtain labels for.

    Output columns:
        - labels (`Union[str, List[str]]`): The label or list of labels for the text.
        - model_name (`str`): The name of the model used to generate the label/s.

    Categories:
        - text-classification

    References:
        - [`Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models`](https://arxiv.org/abs/2408.02442)

    Attributes:
        system_prompt: A prompt to display to the user before the task starts. Contains a default
            message to make the model behave like a classifier specialist.
        n: Number of labels to generate If only 1 is required, corresponds to a label
            classification problem, if >1 it will intend return the "n" labels most representative
            for the text. Defaults to 1.
        context: Context to use when generating the labels. By default contains a generic message,
            but can be used to customize the context for the task.
        examples: List of examples to help the model understand the task, few shots.
        available_labels: List of available labels to choose from when classifying the text, or
            a dictionary with the labels and their descriptions.
        default_label: Default label to use when the text is ambiguous or lacks sufficient information for
            classification. Can be a list in case of multiple labels (n>1).
    """

    system_prompt: Optional[str] = (
        "You are an AI system specialized in generating labels to classify pieces of text. "
        "Your sole purpose is to analyze the given text and provide appropriate classification labels."
    )
    n: PositiveInt = Field(
        default=1,
        description="Number of labels to generate. Defaults to 1.",
    )
    context: Optional[str] = Field(
        default="Generate concise, relevant labels that accurately represent the text's main themes, topics, or categories.",
        description="Context to use when generating the labels.",
    )
    examples: Optional[List[str]] = Field(
        default=None,
        description="List of examples to help the model understand the task, few shots.",
    )
    available_labels: Optional[Union[List[str], Dict[str, str]]] = Field(
        default=None,
        description=(
            "List of available labels to choose from when classifying the text, or "
            "a dictionary with the labels and their descriptions."
        ),
    )
    default_label: Optional[Union[str, List[str]]] = Field(
        default="Unclassified",
        description=(
            "Default label to use when the text is ambiguous or lacks sufficient information for "
            "classification. Can be a list in case of multiple labels (n>1)."
        ),
    )

    _template: Optional[Template] = PrivateAttr(default=None)

    def load(self) -> None:
        super().load()
        self._template = Template(TEXT_CLASSIFICATION_TEMPLATE)
        self._labels_format: str = (
            '"label"'
            if self.n == 1
            else "[" + ", ".join([f'"label_{i}"' for i in range(self.n)]) + "]"
        )
        self._labels_message: str = (
            "Provide the label that best describes the text."
            if self.n == 1
            else f"Provide a list of {self.n} labels that best describe the text."
        )
        self._available_labels_message: str = self._get_available_labels_message()
        self._examples: str = self._get_examples_message()

    def _get_available_labels_message(self) -> str:
        """Prepares the message to display depending on the available labels (if any),
        and whether the labels have a specific context.
        """
        if self.available_labels is None:
            return (
                "Use clear, widely understood terms for labels."
                "Avoid overly specific or obscure labels unless the text demands it."
            )

        msg = (
            "## Labeling the user input\n"
            "Use the available labels to classify the user query{label_context}:\n"
            "available_labels = {available_labels}"
        )
        if isinstance(self.available_labels, list):
            specific_msg = (
                "[\n"
                + indent(
                    "".join([f'"{label}",\n' for label in self.available_labels]),
                    prefix=" " * 4,
                )
                + "]"
            )
            return msg.format(label_context="", available_labels=specific_msg)

        elif isinstance(self.available_labels, dict):
            specific_msg = ""
            for label, description in self.available_labels.items():
                specific_msg += indent(
                    f'"{label}",  # {description}' + "\n", prefix=" " * 4
                )

            specific_msg = "[\n" + specific_msg + "]"
            return msg.format(
                label_context=". Analyze the context of each label specifically",
                available_labels=specific_msg,
            )

    def _get_examples_message(self) -> str:
        """Prepares the message to display depending on the examples provided."""
        if self.examples is None:
            return ""

        examples_msg = "\n".join([f"- {ex}" for ex in self.examples])

        return (
            "\n## Examples\n"
            "Here are some examples to help you understand the task:\n"
            f"{examples_msg}"
        )

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`."""
        return ["text"]

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["labels", "model_name"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        messages = [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    context=f"\n{self.context}",
                    labels_message=self._labels_message,
                    available_labels=self._available_labels_message,
                    examples=self._examples,
                    default_label=self.default_label,
                    labels_format=self._labels_format,
                    text=input["text"],
                ),
            },
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        return self._format_structured_output(output)

    @override
    def get_structured_output(self) -> Dict[str, Any]:
        """Creates the json schema to be passed to the LLM, to enforce generating
        a dictionary with the output which can be directly parsed as a python dictionary.

        Returns:
            JSON Schema of the response to enforce.
        """
        if self.n > 1:

            class MultiLabelSchema(BaseModel):
                labels: List[str]

            schema = MultiLabelSchema.model_json_schema()
        else:

            class SingleLabelSchema(BaseModel):
                labels: str

            schema = SingleLabelSchema.model_json_schema()

        return schema

    def _format_structured_output(
        self, output: str
    ) -> Dict[str, Union[str, List[str]]]:
        """Parses the structured response, which should correspond to a dictionary
        with either `positive`, or `positive` and `negative` keys.

        Args:
            output: The output from the `LLM`.

        Returns:
            Formatted output.
        """
        try:
            return orjson.loads(output)
        except orjson.JSONDecodeError:
            if self.n > 1:
                return {"labels": [None for _ in range(self.n)]}
            return {"labels": None}