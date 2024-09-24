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

import sys
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import orjson as json
from argilla import (
    LabelQuestion,
    MultiLabelQuestion,
    RatingQuestion,
    Record,
    SpanQuestion,
    Suggestion,
    TextField,
    TextQuestion,
)
from argilla._models._settings._questions import (
    LabelQuestionSettings,
    MultiLabelQuestionSettings,
    RatingQuestionSettings,
    SpanQuestionSettings,
    TextQuestionSettings,
)
from jinja2 import Template
from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import override

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput

_LABELQUESTION_SETTINGS = LabelQuestionSettings()
_MULTILABELQUESTION_SETTINGS = MultiLabelQuestionSettings()
_TEXTQUESTION_SETTINGS = TextQuestionSettings()
_RATINGQUESTION_SETTINGS = RatingQuestionSettings(
    options=[
        {"value": 1, "name": "1"},
        {"value": 2, "name": "2"},
    ]
)
_SPANQUESTION_SETTINGS = SpanQuestionSettings()


class ArgillaLabeller(Task):
    """
    Base class for all tasks in ArgiLabel.
    """

    system_prompt: str = (
        "You are an expert annotator and labelling assistant that understands complex domains and natural language processing. "
        "You are given input fields and a question. "
        "You should create a valid QuestionValue JSON object as an answer to the question based on the input fields. "
        "Reason through your response step-by-step."
    )
    question_to_label_instruction: Dict[str, str] = {
        _LABELQUESTION_SETTINGS.type: "Select the appropriate label from the list of provided labels.",
        _MULTILABELQUESTION_SETTINGS.type: "Select none, one or multiple labels from the list of provided labels.",
        _TEXTQUESTION_SETTINGS.type: "Provide a text response to the question.",
        _RATINGQUESTION_SETTINGS.type: "Provide a rating for the question.",
        _SPANQUESTION_SETTINGS.type: "Provide a list of none, one or multiple spans containing of an exact text from the input field value and a label from the list of provided labels.",
    }

    example_records: Optional[
        RuntimeParameter[Optional[List[Union[Dict[str, Any], Record]]]]
    ] = Field(
        default=None,
        description="The few shot example records with responses to be used to answer the question.",
    )
    fields: Optional[List[Union[TextField, Dict[str, Any]]]] = Field(
        default=None,
        description="The field settings for the fields to be used to answer the question.",
    )

    question: Optional[
        RuntimeParameter[
            Union[
                Dict[str, Any],
                LabelQuestion,
                MultiLabelQuestion,
                RatingQuestion,
                SpanQuestion,
                TextQuestion,
                None,
            ]
        ]
    ] = Field(
        default=None,
        description="The question settings for the question to be answered.",
    )

    _template: Union[Template, None] = PrivateAttr(...)
    _client: Optional[Any] = PrivateAttr(None)

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "quality-scorer.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def questions(
        self,
    ) -> List[
        Union[
            Dict[str, Any],
            LabelQuestion,
            MultiLabelQuestion,
            RatingQuestion,
            SpanQuestion,
            TextQuestion,
        ]
    ]:
        return self._questions

    @property
    def inputs(self) -> List[str]:
        return ["records"]

    @property
    def optional_inputs(self) -> List[str]:
        return ["example_records", "fields", "question"]

    def _format_record(
        self, record: Dict[str, Any], fields: List[Dict[str, Any]]
    ) -> str:
        output = []
        for field in fields:
            title = field.get("title", None)
            if title:
                output.append(f"title: {title}")
            description = field.get("description", None)
            if description:
                output.append(f"description: {description}")
            output.append(record.get("fields", {}).get(field.get("name", "")))
        return "\n".join(output)

    def _get_label_instruction(self, question: Dict[str, Any]) -> str:
        question_type = question["settings"]["type"]
        return self.question_to_label_instruction[question_type]

    def _format_question(self, question: Dict[str, Any]) -> str:
        output = []
        output.append(f"title: {question.get('title', '')}")
        output.append(f"description: {question.get('description', '')}")
        output.append(f"label_instruction: {self._get_label_instruction(question)}")
        settings = question.get("settings", {})
        if "options" in settings:
            output.append(
                f"labels: {[option['value'] for option in settings.get('options', [])]}"
            )
        if "allow_overlapping" in settings:
            output.append(f"allow_overlapping: {settings.get('allow_overlapping', '')}")
        return "\n".join(output)

    def _format_example_records(
        self,
        records: List[Dict[str, Any]],
        fields: List[Dict[str, Any]],
        question: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        base = []
        for record in records:
            if record["responses"]:
                base.append(self._format_record(record, fields))
                value = record["responses"][question["name"]][0]["value"]
                formatted_value = self._assign_value_to_question_value_model(
                    value, question
                )
                base.append(f"Response: {formatted_value}")
        return "\n".join(base)

    def format_input(
        self,
        input: Dict[
            str,
            Union[
                Dict[str, Any],
                Record,
                TextField,
                MultiLabelQuestion,
                LabelQuestion,
                RatingQuestion,
                SpanQuestion,
                TextQuestion,
            ],
        ],
    ) -> ChatType:
        record = input["records"]
        fields = input.get("fields", self.fields)
        question = input.get("question", self.question)
        examples = input.get("example_records", self.example_records)
        if any([fields is None, question is None]):
            raise ValueError(
                "Fields and question must be provided during init or through `process` method."
            )
        if not isinstance(record, dict):
            record = record.to_dict()
        if not isinstance(question, dict):
            question = question.serialize()
        fields = [
            field.serialize() if not isinstance(field, dict) else field
            for field in fields
        ]
        if examples:
            examples = [
                example.to_dict() if not isinstance(example, dict) else example
                for example in examples
            ]

        formatted_fields = self._format_record(record, fields)
        formatted_question = self._format_question(question)
        formatted_examples = (
            self._format_example_records(examples, fields, question)
            if examples
            else False
        )
        prompt = self._template.render(
            fields=formatted_fields,
            question=formatted_question,
            examples=formatted_examples,
        )

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @property
    def outputs(self) -> List[str]:
        return ["suggestions"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        fields = input.get("fields", self.fields)
        question = input.get("question", self.question) or self.question
        if not isinstance(question, dict):
            question = question.serialize()
        fields = [
            field.serialize() if not isinstance(field, dict) else field
            for field in fields
        ]

        model = self._get_pydantic_model_of_structured_output(question)
        validated_output = model(**json.loads(output))
        value = self._get_value_from_question_value_model(validated_output)
        if question["settings"]["type"] == _SPANQUESTION_SETTINGS.type:
            value = self._resolve_spans(
                value, input[self.inputs[0]].get("fields", {}).get(fields[0]["name"])
            )
        suggestion = Suggestion(
            value=value,
            question_name=question["name"],
            type="model",
            agent=self.llm.model_name,
        )
        return {self.outputs[0]: suggestion}

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        question = inputs[0].get("question", self.question)
        runtime_parameters = self.llm._runtime_parameters
        if "structured_output" in runtime_parameters:
            warnings.warn(
                "Structured output is handled by ArgillaLabeler internally. Setting structured output to json with schema.",
                stacklevel=2,
            )
        runtime_parameters.update(
            {
                "structured_output": {
                    "format": "json",
                    "schema": self._get_pydantic_model_of_structured_output(question),
                },
            }
        )
        self.llm.set_runtime_parameters(runtime_parameters)
        yield from super().process(inputs)

    def _resolve_spans(
        self, spans: list, field: str, question: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        unique_spans = {span.exact_text: span.label for span in spans}
        formatted_spans = []

        for exact_text, label in unique_spans.items():
            start = field.find(exact_text)
            while start != -1:  # Find all occurrences
                end = start + len(exact_text)
                if label in [
                    option["value"] for option in question["settings"]["options"]
                ]:
                    formatted_spans.append(
                        {
                            "label": label,
                            "start": start,
                            "end": end,
                        }
                    )
                start = field.find(exact_text, start + 1)  # Move to the next occurrence

        return formatted_spans

    def _get_value_from_question_value_model(
        self, question_value_model: BaseModel
    ) -> Any:
        for attr in ["label", "labels", "spans", "rating", "text"]:
            if hasattr(question_value_model, attr):
                return getattr(question_value_model, attr)
        raise ValueError(f"Unsupported question type: {question_value_model}")

    def _assign_value_to_question_value_model(
        self, value: Any, question: Dict[str, Any]
    ) -> BaseModel:
        question_value_model = self._get_pydantic_model_of_structured_output(question)
        for attr in ["label", "labels", "spans", "rating", "text"]:
            try:
                # Initialize the Pydantic model with the attribute and value
                model_dict = {attr: value}
                question_value_model = question_value_model(**model_dict)
                # If initialization is successful, return the model
                return question_value_model.model_dump_json()
            except AttributeError:
                pass
        return value

    def _get_pydantic_model_of_structured_output(
        self,
        question: Dict[str, Any],
    ) -> BaseModel:
        question_type = question["settings"]["type"]
        if question_type == _SPANQUESTION_SETTINGS.type:

            class Span(BaseModel):
                exact_text: str
                label: str

            class QuestionValueModel(BaseModel):
                spans: Optional[List[Span]] = Field(default_factory=list)

        elif question_type == _MULTILABELQUESTION_SETTINGS.type:

            class QuestionValueModel(BaseModel):
                labels: Optional[List[str]] = Field(default_factory=list)

        elif question_type == _LABELQUESTION_SETTINGS.type:

            class QuestionValueModel(BaseModel):
                label: str

        elif question_type == _TEXTQUESTION_SETTINGS.type:

            class QuestionValueModel(BaseModel):
                text: str

        elif question_type == _RATINGQUESTION_SETTINGS.type:

            class QuestionValueModel(BaseModel):
                rating: int
        else:
            raise ValueError(f"Unsupported question type: {question}")

        return QuestionValueModel
