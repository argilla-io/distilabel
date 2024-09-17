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

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from jinja2 import Template
from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import override

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import StepOutput

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

if TYPE_CHECKING:
    from argilla import (
        LabelQuestion,
        MultiLabelQuestion,
        RatingQuestion,
        SpanQuestion,
        TextQuestion,
    )

    from distilabel.steps.typing import StepOutput


class ArgillaLabeller(Task):
    """
    Base class for all tasks in ArgiLabel.
    """

    template_path: Optional[Union[str, Path]] = (
        importlib_resources.files("distilabel")
        / "steps"
        / "tasks"
        / "templates"
        / "argillalabeller.jinja2"
    )

    system_prompt: str = (
        "You are an expert annotator and labelling assistant that understands complex domains and natural language processing. "
        "You are given input fields and a question. "
        "You should create a valid QuestionValue JSON object as an answer to the question based on the input fields. "
        "Reason through your response step-by-step. <think> <reason> <respond>."
    )
    question_to_label_instruction: Dict[str, str] = {
        "LabelQuestion": "Select the appropriate label from the list of provided labels.",
        "MultiLabelQuestion": "Select none, one or multiple labels from the list of provided labels.",
        "TextQuestion": "Provide a text response to the question.",
        "RatingQuestion": "Provide a rating for the question.",
        "SpanQuestion": "Provide a list of none, one or multiple spans containing of an exact text from the input field value and a label from the list of provided labels.",
    }
    settings: RuntimeParameter[Optional[Any]] = Field(
        default=None,
        description="The Argilla settings to be used to answer the question.",
    )
    fields: RuntimeParameter[List[str]] = Field(
        default=None,
        description="The fields to be used to answer the question.",
    )
    question: RuntimeParameter[str] = Field(
        default=None,
        description="The question to be answered.",
    )

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        if isinstance(self.template_path, str):
            self.template_path = Path(self.template_path)

        self._template = Template(open(self.template_path).read())

    @property
    def label_instruction(self) -> str:
        question_type = self.settings.questions[self.question].__class__.__name__
        instruction = self.question_to_label_instruction[question_type]
        return f"The question is {question_type}. {instruction}"

    @property
    def inputs(self) -> List[str]:
        return ["records"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        record = input[self.inputs[0]]
        field_data = [
            {
                "title": self.settings.fields[field].title,
                "description": self.settings.fields[field].description,
                "value": record.fields.get(field),
            }
            for field in self.fields
            if record.fields.get(field) is not None
        ]

        question_instance = self.settings.questions[self.question]
        question_data = {
            "title": question_instance.title,
            "description": question_instance.description,
        }
        if hasattr(question_instance, "labels"):
            question_data["labels"] = question_instance.labels
        if hasattr(question_instance, "allow_overlapping"):
            question_data["allow_overlapping"] = question_instance.allow_overlapping

        prompt = self._template.render(
            fields=field_data,
            question=question_data,
            label_instruction=self.label_instruction,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    @property
    def schema(self) -> BaseModel:
        return self._get_question_value_model(self.settings.questions[self.question])

    @property
    def outputs(self) -> List[str]:
        return ["suggestion"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        from argilla import SpanQuestion, Suggestion

        output = self.schema(**json.loads(output))
        value = self._get_value_from_question_value_model(output)
        if isinstance(self.settings.questions[self.question], SpanQuestion):
            value = self._resolve_spans(value, input["records"].fields[self.fields[0]])
        suggestion = Suggestion(
            value=value,
            question_name=self.question,
            type="model",
            agent=self.llm.model_name,
        )
        return {self.outputs[0]: suggestion}

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        runtime_parameters = self.llm._runtime_parameters
        if "structured_output" in runtime_parameters:
            raise ValueError(
                "Structured output is handled by ArgillaLabeler internally."
            )
        generation_kwargs = runtime_parameters.get("generation_kwargs", {})
        if "temperature" in generation_kwargs:
            raise ValueError("Temperature is handled by ArgillaLabeler internally.")
        generation_kwargs.update({"temperature": 0, "max_new_tokens": 2000})
        runtime_parameters.update(
            {
                "generation_kwargs": generation_kwargs,
                "structured_output": {"format": "json", "schema": self.schema},
            }
        )
        self.llm.set_runtime_parameters(runtime_parameters)
        yield from super().process(inputs)

    def _resolve_spans(self, spans: list, field: str) -> List[Dict[str, Any]]:
        unique_spans = {span.exact_text: span.label for span in spans}
        formatted_spans = []

        for exact_text, label in unique_spans.items():
            start = field.find(exact_text)
            while start != -1:  # Find all occurrences
                end = start + len(exact_text)
                if label in self.settings.questions[self.question].labels:
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

    def _get_question_value_model(
        self,
        question: Union[
            "LabelQuestion",
            "MultiLabelQuestion",
            "SpanQuestion",
            "RatingQuestion",
            "TextQuestion",
        ],
    ) -> BaseModel:
        from argilla import (
            LabelQuestion,
            MultiLabelQuestion,
            RatingQuestion,
            SpanQuestion,
            TextQuestion,
        )

        if isinstance(question, SpanQuestion):

            class Span(BaseModel):
                exact_text: str
                label: str

            class QuestionValueModel(BaseModel):
                spans: List[Span] = []

        elif isinstance(question, MultiLabelQuestion):

            class QuestionValueModel(BaseModel):
                labels: Optional[List[str]] = []

        elif isinstance(question, LabelQuestion):

            class QuestionValueModel(BaseModel):
                label: str

        elif isinstance(question, TextQuestion):

            class QuestionValueModel(BaseModel):
                text: str

        elif isinstance(question, RatingQuestion):

            class QuestionValueModel(BaseModel):
                rating: int

        else:
            raise ValueError(f"Unsupported question type: {question}")

        return QuestionValueModel
