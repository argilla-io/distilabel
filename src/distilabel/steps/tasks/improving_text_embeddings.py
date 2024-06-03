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

import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import override

from distilabel.steps.tasks.base import GeneratorTask, Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import GeneratorStepOutput


class BrainstormEmbeddingTasks(GeneratorTask):
    """Generate text retrieval task descriptions using an `LLM`.

    ...

    References:
        - [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)
    """

    category: Literal[
        "text-retrieval",
        "text-matching-short",
        "text-matching-long",
        "text-classification",
    ]
    flatten_tasks: bool = False

    @property
    def prompt(self) -> ChatType:  # type: ignore
        if self.category == "text-retrieval":
            return [
                {
                    "role": "user",
                    "content": (
                        "Brainstorm a list of potentially useful text retrieval tasks.\n"
                        "Here are a few examples for your reference:\n"
                        " - Provided a scientific claim as query, retrieve documents that help verify or refute the claim.\n"
                        " - Search for documents that answers a FAQ-style query on children's nutrition.\n"
                        "Please adhere to the following guidelines:\n"
                        " - Specify what the query is, and what the desired documents are.\n"
                        " - Each retrieval task should cover a wide range of queries, and should not be too specific.\n"
                        "Your output should always be a python list of strings only, with about 20 elements, and each element "
                        "corresponds to a distinct retrieval task in one sentence. Do not explain yourself or output anything else. "
                        "Be creative!"
                    ),
                },
            ]
        elif self.category == "text-matching-short":
            return [
                {
                    "role": "user",
                    "content": (
                        "Brainstorm a list of text matching tasks where both the queries and the groundtruth documents are very short "
                        "(one or two sentences, even a short phrase).\n"
                        "Here are a few examples:\n"
                        " - Given a scientific paper title, retrieve the title of papers that cite the given paper.\n"
                        " - Match a word with its definition.\n"
                        " - Provided a notable person’s name, identify their occupation or achievement.\n"
                        "Your output must always be a python list of strings only, with about 20 elements, and each element corresponds "
                        "to a distinct task in one sentence. Do not explain yourself or output anything else. Be creative!"
                    ),
                }
            ]
        elif self.category == "text-matching-long":
            return [
                {
                    "role": "user",
                    "content": (
                        "Brainstorm a list of text matching tasks where the queries are long documents.\n"
                        "Here are a few examples:\n"
                        " - Given a document that supports a debatable argument, find another document that contains opposite arguments.\n"
                        " - Provided a lengthy business proposal, retrieve competitive business strategies in the same industry.\n"
                        "Your output must always be a python list of strings only, with about 20 elements, and each element corresponds "
                        "to a distinct task in one sentence. Do not explain yourself or output anything else. Be creative!"
                    ),
                }
            ]
        elif self.category == "text-classification":
            return [
                {
                    "role": "user",
                    "content": (
                        "Brainstorm a list of potentially useful text classification tasks.\n"
                        "Please adhere to the following guidelines:\n"
                        "- Tasks should cover a diverse range of domains and task types.\n"
                        "Your output must always be a python list of strings only, with about 20 elements, and each element corresponds "
                        "to a distinct text classification task in one sentence. Do not explain yourself or output anything else. Be creative!"
                    ),
                }
            ]

    @override
    def process(self, offset: int = 0) -> GeneratorStepOutput:  # type: ignore
        formatted_inputs = [self.prompt]
        outputs = self.llm.generate(
            inputs=formatted_inputs,
            num_generations=self.num_generations,
            **self.llm.generation_kwargs,  # type: ignore
        )

        task_outputs = []
        for input_outputs in outputs:
            formatted_outputs = self._format_outputs(input_outputs)  # type: ignore
            for formatted_output in formatted_outputs:
                if isinstance(formatted_output["tasks"], list) and self.flatten_tasks:
                    task_outputs.extend(
                        [
                            {"task": output, "model_name": self.llm.model_name}
                            for output in formatted_output["tasks"]
                        ]
                    )
                else:
                    task_outputs.append(
                        {
                            "task"
                            if not self.flatten_tasks
                            else "tasks": formatted_output["tasks"],
                            "model_name": self.llm.model_name,
                        }
                    )
        yield task_outputs, True

    @property
    def outputs(self) -> List[str]:
        return ["tasks" if not self.flatten_tasks else "task", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        try:
            if output is not None:
                output = eval(output)
        except Exception:
            pass
        return {"tasks": output}


QUERY_TYPES = ["extremely long-tail", "long-tail", "common"]
QUERY_LENGTH = ["less than 5 words", "5 to 15 words", "at least 10 words"]
DIFFICULTY = ["high school", "college", "PhD"]
CLARITY = ["clear", "understandable with some effort", "ambiguous"]
NUM_WORDS = [50, 100, 200, 300, 400, 500]


class _JSONFormatter(ABC):
    @property
    @abstractmethod
    def keys(self) -> List[str]:
        ...

    @property
    def outputs(self) -> List[str]:
        return self.keys + ["model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        if output is None:
            return {key: None for key in self.keys}

        def escape_backslashes_in_values(s):
            # Regular expression to match the key-value pairs in the dictionary
            pattern = re.compile(r'(".*?":\s*")(.*?)(",?)', re.DOTALL)

            def replace_backslashes(match):
                return (
                    match.group(1)
                    + re.sub(
                        r"(?<!\\)(\n|\r|\t)",
                        r"\\\1",
                        match.group(2),  # .encode("unicode_escape").decode("utf-8"),
                    )
                    + match.group(3)
                )

            return pattern.sub(replace_backslashes, s)

        try:
            output = escape_backslashes_in_values(output)
            output = eval(output)
        except Exception:
            pass

        try:
            pattern = r"```json\n(.*?)```"
            matches = re.findall(pattern, output, re.DOTALL)  # type: ignore
            if matches:
                output = eval(matches[0])
        except Exception:
            pass

        if not isinstance(output, dict):
            return {key: None for key in self.keys}

        return {key: output.get(key, None) for key in self.keys}


class _EmbeddingDataGeneration(_JSONFormatter, Task, ABC):
    seed: int = 42

    def load(self) -> None:
        super().load()
        random.seed(self.seed)

    @property
    def inputs(self) -> List[str]:
        return ["task"]

    @abstractmethod
    def format_input(self, input: Dict[str, Any]) -> ChatType:
        ...


class GenerateTextRetrievalData(_EmbeddingDataGeneration):
    query_type: Optional[Literal["extremely long-tail", "long-tail", "common"]] = None
    query_length: Optional[
        Literal["less than 5 words", "5 to 15 words", "at least 10 words"]
    ] = None
    difficulty: Optional[Literal["high school", "college", "PhD"]] = None
    clarity: Optional[
        Literal["clear", "understandable with some effort", "ambiguous"]
    ] = None
    num_words: Optional[Literal[50, 100, 200, 300, 400, 500]] = None
    language: str = "English"
    """The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf"""

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {
                "role": "user",
                "content": (
                    f"You have been assigned a retrieval task: {input['task']}\n"
                    "Your mission is to write one text retrieval example for this task in JSON format. The JSON object must\n"
                    "contain the following keys:\n"
                    "- 'user_query': a string, a random user search query specified by the retrieval task.\n"
                    "- 'positive_document': a string, a relevant document for the user query.\n"
                    "- 'hard_negative_document': a string, a hard negative document that only appears relevant to the query.\n"
                    "Please adhere to the following guidelines:\n"
                    f"- The 'user_query' should be {self.query_type or random.choice(QUERY_TYPES)}, {self.query_length or random.choice(QUERY_LENGTH)}, {self.clarity or random.choice(CLARITY)}, and diverse in topic.\n"
                    "- All documents must be created independent of the query. Avoid copying the query verbatim. It’s acceptable\n"
                    "if some parts of the 'positive_document' are not topically related to the query.\n"
                    f"- All documents should be at least {self.num_words or random.choice(NUM_WORDS)} words long.\n"
                    "- The 'hard_negative_document' contains some useful information, but it should be less useful or comprehensive compared to the 'positive_document'.\n"
                    f"- Both the query and documents should be in {self.language}.\n"
                    "- Do not provide any explanation in any document on why it is relevant or not relevant to the query.\n"
                    f"- Both the query and documents require {self.difficulty or random.choice(DIFFICULTY)} level education to understand.\n"
                    "Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"
                ).replace("'", '"'),
            },
        ]

    @property
    def keys(self) -> List[str]:
        return [
            "user_query",
            "positive_document",
            "hard_negative_document",
        ]


class GenerateTextMatchingData(_EmbeddingDataGeneration):
    length: Literal["short", "long"]
    language: str = "English"
    """The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf"""

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {
                "role": "user",
                "content": (
                    f"You have been assigned a text matching task: {input['task']}\n"
                    "Your mission is to write one example for this task in JSON format. The JSON object must contain the "
                    "following keys:\n"
                    "- 'input': a string, a random input specified by the task.\n"
                    "- 'positive_document': a string, a relevant document for the 'input' according to the task.\n"
                    "Please adhere to the following guidelines:\n"
                    f"- The values of all fields should be in {self.language}.\n"
                    "- Both the 'input' and 'positive_document' should be very short (a sentence or a phrase), avoid substantial "
                    "word overlaps, otherwise the task would be too easy.\n"
                    "- The 'input' and 'positive_document' should be independent of each other.\n"
                    "Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"
                ).replace("'", '"')
                if self.length == "short"
                else (
                    f"You have been assigned a text matching task: {input['task']}\n"
                    "Your mission is to write one example for this task in JSON format. The JSON object must contain the "
                    "following keys:\n"
                    "- 'input': a string, a random input specified by the task.\n"
                    "- 'positive_document': a string, a relevant document for the 'input' according to the task.\n"
                    "Please adhere to the following guidelines:\n"
                    f"- The values of all fields should be in {self.language}.\n"
                    "- Both the 'input' and 'positive_document' should be long documents (at least 300 words), avoid substantial "
                    "word overlaps, otherwise the task would be too easy.\n"
                    "- The 'input' and 'positive_document' should be independent of each other.\n"
                    "Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"
                ).replace("'", '"'),
            },
        ]

    @property
    def keys(self) -> List[str]:
        return ["input", "positive_document"]


class GenerateTextClassificationData(_EmbeddingDataGeneration):
    difficulty: Optional[Literal["high school", "college", "PhD"]] = None
    clarity: Optional[
        Literal["clear", "understandable with some effort", "ambiguous"]
    ] = None
    language: str = "English"
    """The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf"""

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {
                "role": "user",
                "content": (
                    f"You have been assigned a text classification task: {input['task']}\n"
                    "Your mission is to write one text classification example for this task in JSON format. The JSON object must "
                    "contain the following keys:\n"
                    "- 'input_text': a string, the input text specified by the classification task.\n"
                    "- 'label': a string, the correct label of the input text.\n"
                    "- 'misleading_label': a string, an incorrect label that is related to the task.\n"
                    "Please adhere to the following guidelines:\n"
                    "- The 'input_text' should be diverse in expression.\n"
                    "- The 'misleading_label' must be a valid label for the given task, but not as appropriate as the 'label' for the 'input_text'.\n"
                    f"- The values for all fields should be in {self.language}.\n"
                    "- Avoid including the values of the 'label' and 'misleading_label' fields in the 'input_text', that would make the task too easy.\n"
                    f"- The 'input_text' is {self.clarity or random.choice(CLARITY)} and requires {self.difficulty or random.choice(DIFFICULTY)} level education to comprehend.\n"
                    "Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"
                ).replace("'", '"'),
            },
        ]

    @property
    def keys(self) -> List[str]:
        return ["input_text", "label", "misleading_label"]


class _EmbeddingDataGenerator(_JSONFormatter, GeneratorTask, ABC):
    seed: int = 42

    def load(self) -> None:
        super().load()
        random.seed(self.seed)

    @property
    @abstractmethod
    def prompt(self) -> ChatType:
        ...

    @override
    def process(self, offset: int = 0) -> GeneratorStepOutput:  # type: ignore
        formatted_inputs = [self.prompt]
        outputs = self.llm.generate(
            inputs=formatted_inputs,
            num_generations=self.num_generations,
            **self.llm.generation_kwargs,  # type: ignore
        )

        task_outputs = []
        for input_outputs in outputs:
            formatted_outputs = self._format_outputs(input_outputs)  # type: ignore
            for formatted_output in formatted_outputs:
                task_outputs.append(
                    {
                        **formatted_output,
                        "model_name": self.llm.model_name,
                    }
                )
        yield task_outputs, True


UNITS = ["sentence", "phrase", "passage"]
DIFFICULTIES = ["elementary school", "high school", "college"]
HIGH_SCORES = ["4", "4.5", "5"]
LOW_SCORES = ["2.5", "3", "3.5"]


class MonolingualTripletGenerator(_EmbeddingDataGenerator):
    unit: Optional[Literal["sentence", "phrase", "passage"]] = None
    difficulty: Optional[Literal["elementary school", "high school", "college"]] = None
    high_score: Optional[Literal["4", "4.5", "5"]] = None
    low_score: Optional[Literal["2.5", "3", "3.5"]] = None
    language: str = "English"

    @property
    def prompt(self) -> ChatType:
        return [
            {
                "role": "user",
                "content": (
                    f"Write a {self.unit or random.choice(UNITS)} triple with varying semantic similarity scores in JSON format. The semantic similarity score ranges from 1 to "
                    "5, with 1 denotes least similar and 5 denotes most similar.\n"
                    "Please adhere to the following guidelines:\n"
                    f"- The keys in JSON are 'S1', 'S2', and 'S3', the values are all strings in {self.language}, do not add any other keys.\n"
                    f"- There should be some word overlaps between all three {self.unit or random.choice(UNITS)}s.\n"
                    f"- The similarity score between S1 and S2 should be {self.high_score or random.choice(HIGH_SCORES)}.\n"
                    f"- The similarity score between S1 and S3 should be {self.low_score or random.choice(LOW_SCORES)}.\n"
                    f"- The {self.unit or random.choice(UNITS)}s require {self.difficulty or random.choice(DIFFICULTIES)} level education to understand and should be diverse in terms of topic and length.\n"
                    "Your output must always be a JSON object only with three keys 'S1', 'S2' and 'S3', do not explain yourself or output "
                    "anything else. Be creative!"
                ).replace("'", '"'),
            },
        ]

    @property
    def keys(self) -> List[str]:
        return ["S1", "S2", "S3"]


BITEXT_LOW_SCORES = ["1.5", "2", "2.5"]
BITEXT_HIGH_SCORES = ["4", "4.5", "5"]


class BitextRetrievalGenerator(_EmbeddingDataGenerator):
    unit: Optional[Literal["sentence", "phrase", "passage"]] = None
    difficulty: Optional[Literal["elementary school", "high school", "college"]] = None
    high_score: Optional[Literal["4", "4.5", "5"]] = None
    low_score: Optional[Literal["2.5", "3", "3.5"]] = None
    source_language: str = "English"
    target_language: str = "French"

    @property
    def prompt(self) -> ChatType:
        return [
            {
                "role": "user",
                "content": (
                    f"Write a {self.unit or random.choice(UNITS)} triple with one {self.unit or random.choice(UNITS)} "
                    f"in {self.source_language} and two {self.unit or random.choice(UNITS)}s in {self.target_language} with "
                    "varying translation qualities in JSON format.\n"
                    "The triple is denotes as ('S1', 'S2', 'S3'). The translation quality score ranges from 1 to 5, with higher scores are better.\n"
                    "Please adhere to the following guidelines:\n"
                    f"- The values of 'S1' is a string in {self.source_language}, the value of 'S2' and 'S3' are strings in {self.target_language}.\n"
                    "- There should be some word overlaps between 'S2' and 'S3'.\n"
                    f"- The translation quality score of 'S2' with respect to 'S1' should be {self.high_score or random.choice(BITEXT_HIGH_SCORES)}.\n"
                    f"- The translation quality score of 'S3' with respect to 'S1' should be {self.low_score or random.choice(BITEXT_LOW_SCORES)}.\n"
                    "- 'S3' should be grammatical and fluent, but contain some keyword or number translation errors, or miss some information,\n"
                    "or contain some redundant information.\n"
                    f"- 'S1' requires {self.difficulty or random.choice(DIFFICULTIES)} level education to understand and should be diverse in terms of topic and length.\n"
                    "Your output must always be a JSON object only with three keys 'S1', 'S2' and 'S3', do not explain yourself or output "
                    "anything else. Be creative!"
                ).replace("'", '"'),
            },
        ]

    @property
    def keys(self) -> List[str]:
        return ["S1", "S2", "S3"]
