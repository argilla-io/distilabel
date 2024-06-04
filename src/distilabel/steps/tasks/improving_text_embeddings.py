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
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from jinja2 import Template
from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks.base import GeneratorTask, Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import GeneratorStepOutput


# BASE CLASSES
class _JSONFormatter(ABC):
    @property
    @abstractmethod
    def keys(self) -> List[str]: ...

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

    _template: Union[Template, None] = PrivateAttr(...)
    _template_name: str = PrivateAttr(...)

    def load(self) -> None:
        """Loads the Jinja2 template and sets the random seed."""
        super().load()

        random.seed(self.seed)

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "improving_text_embeddings"
            / self._template_name  # type: ignore
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        return ["task"]

    @abstractmethod
    def format_input(self, input: Dict[str, Any]) -> ChatType: ...


class _EmbeddingDataGenerator(_JSONFormatter, GeneratorTask, ABC):
    seed: int = 42

    _template: Union[Template, None] = PrivateAttr(...)
    _template_name: str = PrivateAttr(...)

    def load(self) -> None:
        """Loads the Jinja2 template and sets the random seed."""
        super().load()

        random.seed(self.seed)

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "improving_text_embeddings"
            / self._template_name  # type: ignore
        )

        self._template = Template(open(_path).read())

    @property
    @abstractmethod
    def prompt(self) -> ChatType: ...

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


# IMPLEMENTED TASKS
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

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "improving_text_embeddings"
            / "brainstorming"
            / self.category
        )

        self._template = Template(open(_path).read())

    @property
    def prompt(self) -> ChatType:  # type: ignore
        return [{"role": "user", "content": self._template.render()}]  # type: ignore

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


QUERY_TYPE = ["extremely long-tail", "long-tail", "common"]
QUERY_LENGTH = ["less than 5 words", "5 to 15 words", "at least 10 words"]
DIFFICULTY = ["high school", "college", "PhD"]
CLARITY = ["clear", "understandable with some effort", "ambiguous"]
NUM_WORDS = [50, 100, 200, 300, 400, 500]


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

    _template_name: str = PrivateAttr(default="text-retrieval")

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    task=input["task"],
                    query_type=self.query_type or random.choice(QUERY_TYPE),
                    query_length=self.query_length or random.choice(QUERY_LENGTH),
                    difficulty=self.difficulty or random.choice(DIFFICULTY),
                    clarity=self.clarity or random.choice(CLARITY),
                    num_words=self.num_words or random.choice(NUM_WORDS),
                    language=self.language,
                ),
            }
        ]

    @property
    def keys(self) -> List[str]:
        return [
            "user_query",
            "positive_document",
            "hard_negative_document",
        ]


class GenerateShortTextMatchingData(_EmbeddingDataGeneration):
    language: str = "English"
    """The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf"""

    _template_name: str = PrivateAttr(default="short-text-matching")

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    task=input["task"],
                    language=self.language,
                ),
            }
        ]

    @property
    def keys(self) -> List[str]:
        return ["input", "positive_document"]


class GenerateLongTextMatchingData(_EmbeddingDataGeneration):
    language: str = "English"
    """The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf"""

    _template_name: str = PrivateAttr(default="long-text-matching")

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    task=input["task"],
                    language=self.language,
                ),
            }
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

    _template_name: str = PrivateAttr(default="text-classification")

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    task=input["task"],
                    difficulty=self.difficulty or random.choice(DIFFICULTY),
                    clarity=self.clarity or random.choice(CLARITY),
                    language=self.language,
                ),
            }
        ]

    @property
    def keys(self) -> List[str]:
        return ["input_text", "label", "misleading_label"]


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

    _template_name: str = PrivateAttr(default="monolingual-triplet")

    @property
    def prompt(self) -> ChatType:
        return [{"role": "user", "content": self._template.render()}]  # type: ignore

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

    _template_name: str = PrivateAttr(default="bitext-retrieval")

    @property
    def prompt(self) -> ChatType:
        return [
            {
                "role": "user",
                "content": self._template.render(
                    unit=self.unit or random.choice(UNITS),
                    difficulty=self.difficulty or random.choice(DIFFICULTIES),
                ),
            }
        ]  # type: ignore

    @property
    def keys(self) -> List[str]:
        return ["S1", "S2", "S3"]
