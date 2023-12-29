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

import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from jinja2 import Template

from distilabel.tasks.base import Task, get_template
from distilabel.tasks.prompt import ChatCompletion, SupportedFormats
from distilabel.utils.argilla import (
    infer_fields_from_dataset_row,
    model_metadata_from_dataset_row,
)
from distilabel.utils.imports import _ARGILLA_AVAILABLE

if _ARGILLA_AVAILABLE:
    import argilla as rg

if TYPE_CHECKING:
    from argilla import FeedbackDataset, FeedbackRecord

_ARGILLA_CHAT_TEMPLATE = get_template("argilla_chat.jinja2")


@dataclass
class Chat:
    messages: Union[str, List[ChatCompletion]]

    def _from_chatml(self, messages: str) -> List[ChatCompletion]:
        pattern = (
            r"\<\|im_start\|\>(\bsystem\b|\buser\b|\bassistant\b)\s+(.*?)\<\|im_end\|\>"
        )
        turns = re.findall(pattern, messages)
        return [ChatCompletion(role=turn[0], content=turn[1]) for turn in turns]

    def _from_llama2(self, messages: str) -> List[ChatCompletion]:
        pattern = r"<s>\[INST\](?: <<SYS>>\n(.*?)\n<</SYS>>\n\n)?(.*?)\s\[\/INST\]\s(.*?)<\/s>"
        turns = re.findall(pattern, messages)
        chat = []
        system_message = False
        for turn in turns:
            for role, content in zip(["system", "user", "assistant"], turn):
                if role == "system" and system_message is True:
                    continue
                if content:
                    chat.append(ChatCompletion(role=role, content=content))  # type: ignore
                elif not content and role == "system" and system_message is False:
                    chat.append(ChatCompletion(role=role, content=""))
                    system_message = True
        return chat

    def _from_zephyr(self, messages: str) -> List[ChatCompletion]:
        pattern = r"<\|(.*?)\|>\n(.*)"
        turns = re.findall(pattern, messages)
        chat = []
        for turn in turns:
            role, message = turn
            if message.endswith("</s>"):
                message = message[:-4]
            chat.append(ChatCompletion(role=role, content=message))
        return chat

    def format_as(  # noqa: C901
        self, input_format: SupportedFormats, output_format: SupportedFormats
    ) -> Union[str, List[ChatCompletion]]:
        # First we convert the `messages` into a list of `ChatCompletion` objects, if necessary.
        # Unless the input and output formats are both the same, so we just need to append the
        # last message to the str or list.
        messages = None
        if input_format == "openai":
            messages = self.messages
        elif input_format == "chatml" and isinstance(self.messages, str):
            messages = self._from_chatml(self.messages)
        elif input_format == "llama2" and isinstance(self.messages, str):
            messages = self._from_llama2(self.messages)
        elif input_format in ["zephyr", "notus"] and isinstance(self.messages, str):
            messages = self._from_zephyr(self.messages)
        else:
            raise ValueError(
                f"`input_format={input_format}` of `type={type(input_format)}` not supported,"
                " please provide a custom `prompt_formatting_fn` or use any of the available"
                f" formats: {SupportedFormats}"
            )

        # Then we format it as the specified output format.
        if output_format == "openai":
            return messages  # type: ignore
        # Re-use the Jinja2 templates from the HuggingFace Hub with defaults replacements.
        else:
            raise ValueError(
                f"`output_format={output_format}` of `type={type(output_format)}` not supported,"
                " please provide a custom `prompt_formatting_fn` or use any of the available"
                f" formats: {SupportedFormats}"
            )


@dataclass
class ChatTask(Task):
    input_format: SupportedFormats
    output_format: SupportedFormats

    __argilla_jinja2_template__: str = _ARGILLA_CHAT_TEMPLATE

    @property
    def input_args_names(self) -> List[str]:
        return ["messages"]

    @property
    def output_args_names(self) -> List[str]:
        return ["generations"]

    def generate_prompt(
        self, messages: Union[str, List[ChatCompletion]]
    ) -> Union[str, List[ChatCompletion]]:
        return Chat(messages=messages).format_as(
            input_format=self.input_format, output_format=self.output_format
        )

    def parse_output(self, output: str) -> Dict[str, str]:
        return {"generations": output}

    def to_argilla_dataset(
        self,
        dataset_row: Dict[str, Any],
        generations_column: Optional[str] = "generations",
    ) -> "FeedbackDataset":
        # First we infer the fields from the input_args_names, but we could also
        # create those manually instead using `rg.TextField(...)`
        fields = [rg.TextField(name="messages", use_markdown=True)]  # type: ignore
        if generations_column is None or generations_column not in dataset_row:
            raise ValueError(
                f"The `generations_column='{generations_column}'` is not present in the dataset"
                f" row. Please provide any of {list(dataset_row.keys())}.",
            )
        fields.append(rg.TextField(name="generations"))  # type: ignore
        remaining_fields = list(
            set(self.input_args_names + self.output_args_names)
            - {"messages", generations_column}
        )
        if len(remaining_fields) > 0:
            fields.extend(
                infer_fields_from_dataset_row(
                    field_names=remaining_fields,
                    dataset_row=dataset_row,
                )
            )
        # Then we add a `LabelQuestion` in order to ask the annotator to assess whether
        # the generated content is correct or not, according to the guidelines defined
        # within the dataset.
        questions = [
            rg.LabelQuestion(  # type: ignore
                name="correctness",
                title="Is the final response correct?",
                labels=["YES", "NO"],
            )
        ]
        # Then we just return the `FeedbackDataset` with the fields, questions, and metadata properties
        # defined above.
        return rg.FeedbackDataset(
            fields=fields,
            questions=questions,  # type: ignore
        )

    def _messages_to_html(self, messages: Union[str, List[ChatCompletion]]) -> str:
        template = Template(open(self.__argilla_jinja2_template__).read())
        if self.input_format != "openai":
            messages = Chat(messages=messages).format_as(
                input_format=self.input_format, output_format="openai"
            )
        return template.render(messages=messages).strip()

    def to_argilla_record(self, dataset_row: Dict[str, Any]) -> "FeedbackRecord":
        """Converts a dataset row to an Argilla `FeedbackRecord`."""
        fields = {"messages": self._messages_to_html(messages=dataset_row["messages"])}
        remaining_args = list(
            set(self.input_args_names + self.output_args_names) - {"messages"}
        )
        for arg_name in remaining_args:
            arg_value = dataset_row[arg_name]
            if isinstance(arg_value, list):
                for idx, value in enumerate(arg_value, start=1):
                    value = (
                        value.strip()
                        if isinstance(value, str)
                        else "\n".join(value)
                        if isinstance(value, list)
                        else ""
                    )
                    fields[f"{arg_name}-{idx}"] = value
            elif isinstance(arg_value, str):
                fields[arg_name] = arg_value.strip() if arg_value else ""
            else:
                warnings.warn(
                    f"Unsupported input type ({type(arg_value)}), skipping...",
                    UserWarning,
                    stacklevel=2,
                )
        # Then we add the model metadata from the `generation_model` and `labelling_model`
        # columns of the dataset, if they exist.
        metadata = model_metadata_from_dataset_row(dataset_row=dataset_row)
        # Finally, we return the `FeedbackRecord` with the fields and the metadata
        return rg.FeedbackRecord(fields=fields, metadata=metadata)  # type: ignore
