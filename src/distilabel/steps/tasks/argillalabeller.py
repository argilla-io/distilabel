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
    Annotate Argilla records based on input fields, example records and question settings.

    This task is designed to facilitate the annotation of Argilla records by leveraging a pre-trained LLM.
    It uses a system prompt that guides the LLM to understand the input fields, the question type,
    and the question settings. The task then formats the input data and generates a response based on the question.
    The response is validated against the question's value model, and the final suggestion is prepared for annotation.

    Attributes:
        _template: a Jinja2 template used to format the input for the LLM.

    Input columns:
        - record (`Record`): The record to be annotated.
        - fields (`Optional[List[Dict[str, Any]]]`): The list of field settings for the input fields.
        - question (`Optional[Dict[str, Any]]`): The question settings for the question to be answered.
        - example_records (`Optional[List[Dict[str, Any]]]`): The few shot example records with responses to be used to answer the question.
        - guidelines (`Optional[str]`): The guidelines for the annotation task.

    Output columns:
        - suggestions (`Dict[str, Any]`): The final suggestion for annotation.

    Categories:
        - text-classification
        - scorer
        - text-generation

    References:
        - [`Argilla: Argilla is a collaboration tool for AI engineers and domain experts to build high-quality datasets`](https://github.com/argilla-io/argilla/)

    Examples:
        Annotate a record with the same dataset and question:

        ```python
        import argilla as rg
        from distilabel.steps.tasks import ArgillaLabeller
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        labeller = ArgillaLabeller(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            )
            fields=[rg.TextField(name="text")],
            question=rg.LabelQuestion(name="label", options=["positive", "negative"]),
            example_records=[rg.Record(fields={"text": "This is a test example record"}, responses={"label": rg.Response(question_name="label", value="positive")})],
            guidelines="These are the guidelines for the annotation task."
        )

        labeller.load()

        result = next(
            labeller.process(
                [
                    {
                        "record": rg.Record(fields={"text": "This is a test record"}).to_dict(),
                    }
                ]
            )
        )
        ```

        Annotate a record with alternating datasets and questions:

        ```python
        import argilla as rg
        from distilabel.steps.tasks import ArgillaLabeller
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        labeller = ArgillaLabeller(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            )
        )

        labeller.load()

        result = next(
            labeller.process(
                [
                    {
                        "record": rg.Record(fields={"text": "This is a test record"}).to_dict(),
                        "fields": [rg.TextField(name="text")],
                        "question": rg.LabelQuestion(name="label", options=["positive", "negative"]),
                        "example_records": [
                            rg.Record(
                                fields={"text": "This is a test example record"},
                                responses={"label": rg.Response(question_name="label", value="positive")}
                            )
                        ],
                        "guidelines": "These are the guidelines for the annotation task."
                    }
                ]
            )
        )
        ```

        Overwrite default prompts and instructions:

        ```python
        import argilla as rg
        from distilabel.steps.tasks import ArgillaLabeller
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        labeller = ArgillaLabeller(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            ),
            system_prompt="You are an expert annotator and labelling assistant that understands complex domains and natural language processing.",
            question_to_label_instruction={
                "label": "Select the appropriate label from the list of provided labels.",
                "multilabel": "Select none, one or multiple labels from the list of provided labels.",
                "text": "Provide a text response to the question.",
                "rating": "Provide a rating for the question.",
                "span": "Provide a list of none, one or multiple spans containing of an exact text from the input field value and a label from the list of provided labels.",
            },
        )

        labeller.load()
        ```
    """

    system_prompt: str = (
        "You are an expert annotator and labelling assistant that understands complex domains and natural language processing. "
        "You are given input fields and a question. "
        "You should create a valid JSON object as an answer to the question based on the input fields. "
        "1. Understand the input fields. "
        "2. Understand the question type and the question settings. "
        "3. Reason through your response step-by-step. "
        "4. Provide a valid JSON object as an answer to the question."
    )
    question_to_label_instruction: Dict[str, str] = {
        _LABELQUESTION_SETTINGS.type: "Select the appropriate label from the list of provided labels.",
        _MULTILABELQUESTION_SETTINGS.type: "Select none, one or multiple labels from the list of provided labels.",
        _TEXTQUESTION_SETTINGS.type: "Provide a text response to the question.",
        _RATINGQUESTION_SETTINGS.type: "Provide a rating for the question.",
        _SPANQUESTION_SETTINGS.type: "Provide a list of none, one or multiple spans containing of an exact text from the input field value and a label from the list of provided labels.",
    }
    example_records: Optional[
        RuntimeParameter[Union[List[Union[Dict[str, Any], Record]], None]]
    ] = Field(
        default=None,
        description="The few shot example records with responses to be used to answer the question.",
    )
    fields: Optional[
        RuntimeParameter[Union[List[Union[TextField, Dict[str, Any]]], None]]
    ] = Field(
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
    guidelines: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The guidelines for the annotation task.",
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
            / "argillalabeller.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        return ["record"]

    @property
    def optional_inputs(self) -> List[str]:
        return ["fields", "question", "example_records", "guidelines"]

    def _format_record(
        self, record: Dict[str, Any], fields: List[Dict[str, Any]]
    ) -> str:
        """Format the record fields into a string.

        Args:
            record (Dict[str, Any]): The record to format.
            fields (List[Dict[str, Any]]): The fields to format.

        Returns:
            str: The formatted record fields.
        """
        output = []
        for field in fields:
            if title := field.get("title"):
                output.append(f"title: {title}")
            if description := field.get("description"):
                output.append(f"description: {description}")
            output.append(record.get("fields", {}).get(field.get("name", "")))
        return "\n".join(output)

    def _get_label_instruction(self, question: Dict[str, Any]) -> str:
        """Get the label instruction for the question.

        Args:
            question (Dict[str, Any]): The question to get the label instruction for.

        Returns:
            str: The label instruction for the question.
        """
        question_type = question["settings"]["type"]
        return self.question_to_label_instruction[question_type]

    def _format_question(self, question: Dict[str, Any]) -> str:
        """Format the question settings into a string.

        Args:
            question (Dict[str, Any]): The question to format.

        Returns:
            str: The formatted question.
        """
        output = [
            f"title: {question.get('title', '')}",
            f"description: {question.get('description', '')}",
            f"label_instruction: {self._get_label_instruction(question)}",
        ]
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
    ) -> str:
        """Format the example records into a string.

        Args:
            records (List[Dict[str, Any]]): The records to format.
            fields (List[Dict[str, Any]]): The fields to format.
            question (Dict[str, Any]): The question to format.

        Returns:
            str: The formatted example records.
        """
        base = []
        for record in records:
            responses = record.get("responses", {})
            if responses.get(question["name"]):
                base.append(self._format_record(record, fields))
                value = responses[question["name"]][0]["value"]
                formatted_value = self._assign_value_to_question_value_model(
                    value, question
                )
                base.append(f"Response: {formatted_value}")
                base.append("")
            else:
                warnings.warn(
                    f"Record {record} has no response for question {question['name']}. Skipping example record.",
                    stacklevel=2,
                )
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
        """Format the input into a chat message.

        Args:
            input (Dict[str, Union[Dict[str, Any], Record, TextField, MultiLabelQuestion, LabelQuestion, RatingQuestion, SpanQuestion, TextQuestion]]): The input to format.

        Returns:
            ChatType: The formatted chat message.
        """
        record = input[self.inputs[0]]
        fields = input.get(self.optional_inputs[0], self.fields)
        question = input.get(self.optional_inputs[1], self.question)
        examples = input.get(self.optional_inputs[2], self.example_records)
        guidelines = input.get(self.optional_inputs[3], self.guidelines)

        if any([fields is None, question is None]):
            raise ValueError(
                "Fields and question must be provided during init or through `process` method."
            )

        record = record.to_dict() if not isinstance(record, dict) else record
        question = question.serialize() if not isinstance(question, dict) else question
        fields = [
            field.serialize() if not isinstance(field, dict) else field
            for field in fields
        ]
        examples = (
            [
                example.to_dict() if not isinstance(example, dict) else example
                for example in examples
            ]
            if examples
            else None
        )

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
            guidelines=guidelines,
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
        """Format the output into a dictionary.

        Args:
            output (Union[str, None]): The output to format.
            input (Dict[str, Any]): The input to format.

        Returns:
            Dict[str, Any]: The formatted output.
        """
        fields = input.get("fields", self.fields)
        question = input.get("question", self.question) or self.question
        question = question.serialize() if not isinstance(question, dict) else question
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
        ).serialize()
        return {self.outputs[0]: suggestion}

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        """Process the input through the task.

        Args:
            inputs (StepInput): The input to process.

        Returns:
            StepOutput: The output of the task.
        """
        questions = [input.get("question", self.question) for input in inputs]
        questions = [
            question.serialize() if not isinstance(question, dict) else question
            for question in questions
        ]
        if not all(question == questions[0] for question in questions):
            warnings.warn(
                "Not all questions are the same. Processing each question separately by setting the structured output for each question. This may impact performance.",
                stacklevel=2,
            )
            for input, question in zip(inputs, questions):
                runtime_parameters = self.llm._runtime_parameters
                runtime_parameters.update(
                    {
                        "structured_output": {
                            "format": "json",
                            "schema": self._get_pydantic_model_of_structured_output(
                                question
                            ),
                        },
                    }
                )
                self.llm.set_runtime_parameters(runtime_parameters)
                yield from super().process([input])
        else:
            question = questions[0]
            runtime_parameters = self.llm._runtime_parameters
            runtime_parameters.update(
                {
                    "structured_output": {
                        "format": "json",
                        "schema": self._get_pydantic_model_of_structured_output(
                            question
                        ),
                    },
                }
            )
            self.llm.set_runtime_parameters(runtime_parameters)
            yield from super().process(inputs)

    def _resolve_spans(
        self, spans: list, field: str, question: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Resolve the spans into a list of dictionaries.

        Args:
            spans (list): The spans to resolve.
            field (str): The field to resolve the spans for.
            question (Dict[str, Any]): The question to resolve the spans for.

        Returns:
            List[Dict[str, Any]]: The resolved spans.
        """
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
        """Get the value from the question value model.

        Args:
            question_value_model (BaseModel): The question value model to get the value from.

        Returns:
            Any: The value from the question value model.
        """
        for attr in ["label", "labels", "spans", "rating", "text"]:
            if hasattr(question_value_model, attr):
                return getattr(question_value_model, attr)
        raise ValueError(f"Unsupported question type: {question_value_model}")

    def _assign_value_to_question_value_model(
        self, value: Any, question: Dict[str, Any]
    ) -> BaseModel:
        """Assign the value to the question value model.

        Args:
            value (Any): The value to assign.
            question (Dict[str, Any]): The question to assign the value to.

        Returns:
            BaseModel: The question value model with the assigned value.
        """
        question_value_model = self._get_pydantic_model_of_structured_output(question)
        for attr in ["label", "labels", "spans", "rating", "text"]:
            try:
                model_dict = {attr: value}
                question_value_model = question_value_model(**model_dict)
                return question_value_model.model_dump_json()
            except AttributeError:
                pass
        return value

    def _get_pydantic_model_of_structured_output(
        self,
        question: Dict[str, Any],
    ) -> BaseModel:
        """Get the Pydantic model of the structured output.

        Args:
            question (Dict[str, Any]): The question to get the Pydantic model of the structured output for.

        Returns:
            BaseModel: The Pydantic model of the structured output.
        """
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
