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
from pydantic import Field, PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks.base import GeneratorTask, Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import GeneratorStepOutput


# BASE CLASSES
class _JSONFormatter(ABC):
    """Abstract class that sets the `outputs` property and `format_output` method, assuming
    that the output is a JSON string with the keys specified in the `keys` property. So on,
    this class is intended to be used whenever we get a JSON string as the `LLM` output with
    a set of `keys` we know are there.

    Note:
        At the moment this abstract class is only intended to be used for the tasks defined
        below based on the output generated by those. Also note that this is not a replacement
        for neither the `StructuredGeneration` task nor for the `structured_output` argument
        of an `LLM` subclass.
    """

    @property
    @abstractmethod
    def keys(self) -> List[str]:
        """Contains the `keys` that will be parsed from the `LLM` output into a Python dict."""
        ...

    @property
    def outputs(self) -> List[str]:
        """Contains the output columns produced by the `process` method of the task. In this
        case, it consists of the `keys` (i.e. the JSON keys) and the `model_name`.
        """
        return self.keys + ["model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """Method to parse the JSON output into a Python dictionary based on the `keys` property.

        Args:
            output: The JSON string output produced by the `LLM`.
            input: The input dictionary that was used to generate the output.

        Returns:
            A Python dictionary with the parsed output based on the `keys` property.
        """
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
    """Base class for the subtasks related to embedding data generation as presented in the
    paper "Improving Text Embeddings with Large Language Models", including a pre-defined `load`
    method to load a Jinja2 template based on the `_template_name` private attribute (to be set
    in each of the subclasses), assuming that the `inputs` property only expects the `task`, while
    keeping the `format_input` as an abstract method to be implemented in the subclasses.

    Attributes:
        seed: The random seed to be set in case there's any sampling within the `format_input` method.
        _template: The Jinja2 template to be rendered within the `format_input` method with the
            provided arguments.
        _template_name: The name of the Jinja2 template file within the
            `distilabel/steps/tasks/templates/improving_text_embeddings` directory.
    """

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
            / f"{self._template_name}.jinja2"  # type: ignore
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        """Contains the input columns expected by the `process` method of the task. In this
        case, it consists of the `task`; ideally produced in a previous task which should be
        preferrably `EmbeddingTaskGenerator` (as per the original implementation)."""
        return ["task"]


class _EmbeddingDataGenerator(_JSONFormatter, GeneratorTask, ABC):
    """Base class for the subtasks related to embedding data generation as presented in the
    paper "Improving Text Embeddings with Large Language Models" that generate data without
    an input i.e. `GeneratorStep` or `GeneratorTask`. This class includes a pre-defined `load`
    method to load a Jinja2 template based on the `_template_name` private attribute (to be set
    in each of the subclasses), assuming that the `prompt` property only expects the `task`, while
    keeping the `format_input` as an abstract method to be implemented in the subclasses.

    Attributes:
        seed: The random seed to be set in case there's any sampling within the `format_input` method.
        _template: The Jinja2 template to be rendered within the `format_input` method with the
            provided arguments.
        _template_name: The name of the Jinja2 template file within the
            `distilabel/steps/tasks/templates/improving_text_embeddings` directory.
    """

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
            / f"{self._template_name}.jinja2"  # type: ignore
        )

        self._template = Template(open(_path).read())

    @property
    @abstractmethod
    def prompt(self) -> ChatType:
        """The prompt to be used for the generation step, ideally rendering the `_template`."""
        ...

    @override
    def process(self, offset: int = 0) -> GeneratorStepOutput:  # type: ignore
        """Method to run the `LLM` generation with the `prompt`, as well as formatting the
        outputs accordingly for the task i.e. via the `_JSONFormatter` inheritance. So on, the
        `LLM` ideally will be prompted to produce JSON content and then the `format_output`
        method will parse it into a Python dictionary based on the `keys` property.

        Args:
            offset: The offset to start the generation from. Defaults to 0.

        Yields:
            The output rows and a boolean indicating if it's the last batch or not.
        """
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
class EmbeddingTaskGenerator(GeneratorTask):
    """Generate task descriptions for embedding-related tasks using an `LLM`.

    `EmbeddingTaskGenerator` is a `GeneratorTask` that doesn't receieve any input besides the
    provided attributes that generates task descriptions for embedding-related tasks using a
    pre-defined prompt based on the `category` attribute. The `category` attribute should be
    one of the following:

    - `text-retrieval`: Generate task descriptions for text retrieval tasks.
    - `text-matching-short`: Generate task descriptions for short text matching tasks.
    - `text-matching-long`: Generate task descriptions for long text matching tasks.
    - `text-classification`: Generate task descriptions for text classification tasks.

    Attributes:
        category: The category of the task to be generated, which can either be `text-retrieval`,
            `text-matching-short`, `text-matching-long`, or `text-classification`.
        flatten_tasks: Whether to flatten the tasks i.e. since a list of tasks is generated by the
            `LLM`, this attribute indicates whether to flatten the list or not. Defaults to `False`,
            meaning that running this task with `num_generations=1` will return a `distilabel.Distiset`
            with one row only containing a list with around 20 tasks; otherwise, if set to `True`, it
            will return a `distilabel.Distiset` with around 20 rows, each containing one task.

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
            / f"{self.category}.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def prompt(self) -> ChatType:  # type: ignore
        """The prompt to be used in the `process` method, rendering the `_template` with the
        provided args / attributes.
        """
        return [{"role": "user", "content": self._template.render().strip()}]  # type: ignore

    @override
    def process(self, offset: int = 0) -> GeneratorStepOutput:  # type: ignore
        """Method to run the `LLM` generation with the `prompt`, as well as formatting the
        outputs accordingly for the task i.e. via the `_JSONFormatter` inheritance. So on, the
        `LLM` ideally will be prompted to produce JSON content and then the `format_output`
        method will parse it into a Python dictionary based on the `keys` property.

        Args:
            offset: The offset to start the generation from. Defaults to 0.

        Yields:
            The output rows and a boolean indicating if it's the last batch or not.
        """
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
                            (
                                "task" if not self.flatten_tasks else "tasks"
                            ): formatted_output["tasks"],
                            "model_name": self.llm.model_name,
                        }
                    )
        yield task_outputs, True

    @property
    def outputs(self) -> List[str]:
        """Contains the output columns produced by the `process` method of the task. In this
        case, it consists of the `tasks` or `task` (depending on the `flatten_tasks` attribute)
        and the `model_name`.
        """
        return ["tasks" if not self.flatten_tasks else "task", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """Method to parse the JSON output into a Python dictionary based on the `keys` property.

        Args:
            output: The JSON string output produced by the `LLM`.
            input: The input dictionary that was used to generate the output.

        Returns:
            A Python dictionary with the parsed output based on the `keys` property.
        """
        try:
            if output is not None:
                output = eval(output)
        except Exception:
            pass
        return {"tasks": output}


class GenerateTextRetrievalData(_EmbeddingDataGeneration):
    """Generate text retrieval data with an `LLM` to later on train an embedding model.

    `GenerateTextRetrievalData` is a `Task` that generates text retrieval data with an
    `LLM` to later on train an embedding model. The task is based on the paper "Improving
    Text Embeddings with Large Language Models" and the data is generated based on the
    provided attributes, or randomly sampled if not provided.

    Attributes:
        language: The language of the data to be generated, which can be any of the languages
            retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.
        query_type: The type of query to be generated, which can be `extremely long-tail`, `long-tail`,
            or `common`. Defaults to `None`, meaning that it will be randomly sampled.
        query_length: The length of the query to be generated, which can be `less than 5 words`, `5 to 15 words`,
            or `at least 10 words`. Defaults to `None`, meaning that it will be randomly sampled.
        difficulty: The difficulty of the query to be generated, which can be `high school`, `college`, or `PhD`.
            Defaults to `None`, meaning that it will be randomly sampled.
        clarity: The clarity of the query to be generated, which can be `clear`, `understandable with some effort`,
            or `ambiguous`. Defaults to `None`, meaning that it will be randomly sampled.
        num_words: The number of words in the query to be generated, which can be `50`, `100`, `200`, `300`, `400`, or `500`.
            Defaults to `None`, meaning that it will be randomly sampled.
        seed: The random seed to be set in case there's any sampling within the `format_input` method.
    """

    language: str = Field(
        default="English",
        description="The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf",
    )

    query_type: Optional[Literal["extremely long-tail", "long-tail", "common"]] = None
    query_length: Optional[
        Literal["less than 5 words", "5 to 15 words", "at least 10 words"]
    ] = None
    difficulty: Optional[Literal["high school", "college", "PhD"]] = None
    clarity: Optional[
        Literal["clear", "understandable with some effort", "ambiguous"]
    ] = None
    num_words: Optional[Literal[50, 100, 200, 300, 400, 500]] = None

    _template_name: str = PrivateAttr(default="text-retrieval")

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """Method to format the input based on the `task` and the provided attributes, or just
        randomly sampling those if not provided. This method will render the `_template` with
        the provided arguments and return an OpenAI formatted chat i.e. a `ChatType`, assuming that
        there's only one turn, being from the user with the content being the rendered `_template`.

        Args:
            input: The input dictionary containing the `task` to be used in the `_template`.

        Returns:
            A list with a single chat containing the user's message with the rendered `_template`.
        """
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    task=input["task"],
                    language=self.language,
                    query_type=self.query_type
                    or random.choice(["extremely long-tail", "long-tail", "common"]),
                    query_length=self.query_length
                    or random.choice(
                        ["less than 5 words", "5 to 15 words", "at least 10 words"]
                    ),
                    difficulty=self.difficulty
                    or random.choice(["high school", "college", "PhD"]),
                    clarity=self.clarity
                    or random.choice(
                        ["clear", "understandable with some effort", "ambiguous"]
                    ),
                    num_words=self.num_words
                    or random.choice([50, 100, 200, 300, 400, 500]),
                ).strip(),
            }
        ]

    @property
    def keys(self) -> List[str]:
        """Contains the `keys` that will be parsed from the `LLM` output into a Python dict."""
        return [
            "user_query",
            "positive_document",
            "hard_negative_document",
        ]


class GenerateShortTextMatchingData(_EmbeddingDataGeneration):
    """Generate short text matching data with an `LLM` to later on train an embedding model.

    `GenerateShortTextMatchingData` is a `Task` that generates short text matching data with an
    `LLM` to later on train an embedding model. The task is based on the paper "Improving
    Text Embeddings with Large Language Models" and the data is generated based on the
    provided attributes, or randomly sampled if not provided.

    Attributes:
        language: The language of the data to be generated, which can be any of the languages
            retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.
        seed: The random seed to be set in case there's any sampling within the `format_input` method.
    """

    language: str = Field(
        default="English",
        description="The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf",
    )

    _template_name: str = PrivateAttr(default="short-text-matching")

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """Method to format the input based on the `task` and the provided attributes, or just
        randomly sampling those if not provided. This method will render the `_template` with
        the provided arguments and return an OpenAI formatted chat i.e. a `ChatType`, assuming that
        there's only one turn, being from the user with the content being the rendered `_template`.

        Args:
            input: The input dictionary containing the `task` to be used in the `_template`.

        Returns:
            A list with a single chat containing the user's message with the rendered `_template`.
        """
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    task=input["task"],
                    language=self.language,
                ).strip(),
            }
        ]

    @property
    def keys(self) -> List[str]:
        """Contains the `keys` that will be parsed from the `LLM` output into a Python dict."""
        return ["input", "positive_document"]


class GenerateLongTextMatchingData(_EmbeddingDataGeneration):
    """Generate long text matching data with an `LLM` to later on train an embedding model.

    `GenerateLongTextMatchingData` is a `Task` that generates long text matching data with an
    `LLM` to later on train an embedding model. The task is based on the paper "Improving
    Text Embeddings with Large Language Models" and the data is generated based on the
    provided attributes, or randomly sampled if not provided.

    Attributes:
        language: The language of the data to be generated, which can be any of the languages
            retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.
        seed: The random seed to be set in case there's any sampling within the `format_input` method.
    """

    language: str = Field(
        default="English",
        description="The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf",
    )

    _template_name: str = PrivateAttr(default="long-text-matching")

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """Method to format the input based on the `task` and the provided attributes, or just
        randomly sampling those if not provided. This method will render the `_template` with
        the provided arguments and return an OpenAI formatted chat i.e. a `ChatType`, assuming that
        there's only one turn, being from the user with the content being the rendered `_template`.

        Args:
            input: The input dictionary containing the `task` to be used in the `_template`.

        Returns:
            A list with a single chat containing the user's message with the rendered `_template`.
        """
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    task=input["task"],
                    language=self.language,
                ).strip(),
            }
        ]

    @property
    def keys(self) -> List[str]:
        """Contains the `keys` that will be parsed from the `LLM` output into a Python dict."""
        return ["input", "positive_document"]


class GenerateTextClassificationData(_EmbeddingDataGeneration):
    """Generate text classification data with an `LLM` to later on train an embedding model.

    `GenerateTextClassificationData` is a `Task` that generates text classification data with an
    `LLM` to later on train an embedding model. The task is based on the paper "Improving
    Text Embeddings with Large Language Models" and the data is generated based on the
    provided attributes, or randomly sampled if not provided.

    Attributes:
        language: The language of the data to be generated, which can be any of the languages
            retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.
        difficulty: The difficulty of the query to be generated, which can be `high school`, `college`, or `PhD`.
            Defaults to `None`, meaning that it will be randomly sampled.
        clarity: The clarity of the query to be generated, which can be `clear`, `understandable with some effort`,
            or `ambiguous`. Defaults to `None`, meaning that it will be randomly sampled.
        seed: The random seed to be set in case there's any sampling within the `format_input` method.
    """

    language: str = Field(
        default="English",
        description="The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf",
    )

    difficulty: Optional[Literal["high school", "college", "PhD"]] = None
    clarity: Optional[
        Literal["clear", "understandable with some effort", "ambiguous"]
    ] = None

    _template_name: str = PrivateAttr(default="text-classification")

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """Method to format the input based on the `task` and the provided attributes, or just
        randomly sampling those if not provided. This method will render the `_template` with
        the provided arguments and return an OpenAI formatted chat i.e. a `ChatType`, assuming that
        there's only one turn, being from the user with the content being the rendered `_template`.

        Args:
            input: The input dictionary containing the `task` to be used in the `_template`.

        Returns:
            A list with a single chat containing the user's message with the rendered `_template`.
        """
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    task=input["task"],
                    language=self.language,
                    difficulty=self.difficulty
                    or random.choice(["high school", "college", "PhD"]),
                    clarity=self.clarity
                    or random.choice(
                        ["clear", "understandable with some effort", "ambiguous"]
                    ),
                ).strip(),
            }
        ]

    @property
    def keys(self) -> List[str]:
        """Contains the `keys` that will be parsed from the `LLM` output into a Python dict."""
        return ["input_text", "label", "misleading_label"]


class MonolingualTripletGenerator(_EmbeddingDataGenerator):
    """Generate monolingual triplets with an `LLM` to later on train an embedding model.

    `MonolingualTripletGenerator` is a `GeneratorTask` that generates monolingual triplets with an
    `LLM` to later on train an embedding model. The task is based on the paper "Improving
    Text Embeddings with Large Language Models" and the data is generated based on the
    provided attributes, or randomly sampled if not provided.

    Attributes:
        language: The language of the data to be generated, which can be any of the languages
            retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.
        unit: The unit of the data to be generated, which can be `sentence`, `phrase`, or `passage`.
            Defaults to `None`, meaning that it will be randomly sampled.
        difficulty: The difficulty of the query to be generated, which can be `elementary school`, `high school`, or `college`.
            Defaults to `None`, meaning that it will be randomly sampled.
        high_score: The high score of the query to be generated, which can be `4`, `4.5`, or `5`.
            Defaults to `None`, meaning that it will be randomly sampled.
        low_score: The low score of the query to be generated, which can be `2.5`, `3`, or `3.5`.
            Defaults to `None`, meaning that it will be randomly sampled.
        seed: The random seed to be set in case there's any sampling within the `format_input` method.
    """

    language: str = Field(
        default="English",
        description="The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf",
    )

    unit: Optional[Literal["sentence", "phrase", "passage"]] = None
    difficulty: Optional[Literal["elementary school", "high school", "college"]] = None
    high_score: Optional[Literal["4", "4.5", "5"]] = None
    low_score: Optional[Literal["2.5", "3", "3.5"]] = None

    _template_name: str = PrivateAttr(default="monolingual-triplet")

    @property
    def prompt(self) -> ChatType:
        """Contains the `prompt` to be used in the `process` method, rendering the `_template`; and
        formatted as an OpenAI formatted chat i.e. a `ChatType`, assuming that there's only one turn,
        being from the user with the content being the rendered `_template`.
        """
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    language=self.language,
                    unit=self.unit or random.choice(["sentence", "phrase", "passage"]),
                    difficulty=self.difficulty
                    or random.choice(["elementary school", "high school", "college"]),
                    high_score=self.high_score or random.choice(["4", "4.5", "5"]),
                    low_score=self.low_score or random.choice(["2.5", "3", "3.5"]),
                ).strip(),
            }
        ]  # type: ignore

    @property
    def keys(self) -> List[str]:
        """Contains the `keys` that will be parsed from the `LLM` output into a Python dict."""
        return ["S1", "S2", "S3"]


class BitextRetrievalGenerator(_EmbeddingDataGenerator):
    """Generate bitext retrieval data with an `LLM` to later on train an embedding model.

    `BitextRetrievalGenerator` is a `GeneratorTask` that generates bitext retrieval data with an
    `LLM` to later on train an embedding model. The task is based on the paper "Improving
    Text Embeddings with Large Language Models" and the data is generated based on the
    provided attributes, or randomly sampled if not provided.

    Attributes:
        source_language: The source language of the data to be generated, which can be any of the languages
            retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.
        target_language: The target language of the data to be generated, which can be any of the languages
            retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf.
        unit: The unit of the data to be generated, which can be `sentence`, `phrase`, or `passage`.
            Defaults to `None`, meaning that it will be randomly sampled.
        difficulty: The difficulty of the query to be generated, which can be `elementary school`, `high school`, or `college`.
            Defaults to `None`, meaning that it will be randomly sampled.
        high_score: The high score of the query to be generated, which can be `4`, `4.5`, or `5`.
            Defaults to `None`, meaning that it will be randomly sampled.
        low_score: The low score of the query to be generated, which can be `2.5`, `3`, or `3.5`.
            Defaults to `None`, meaning that it will be randomly sampled.
        seed: The random seed to be set in case there's any sampling within the `format_input` method.
    """

    source_language: str = Field(
        default="English",
        description="The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf",
    )
    target_language: str = Field(
        default=...,
        description="The languages are retrieved from the list of XLM-R in the Appendix A of https://aclanthology.org/2020.acl-main.747.pdf",
    )

    unit: Optional[Literal["sentence", "phrase", "passage"]] = None
    difficulty: Optional[Literal["elementary school", "high school", "college"]] = None
    high_score: Optional[Literal["4", "4.5", "5"]] = None
    low_score: Optional[Literal["2.5", "3", "3.5"]] = None

    _template_name: str = PrivateAttr(default="bitext-retrieval")

    @property
    def prompt(self) -> ChatType:
        """Contains the `prompt` to be used in the `process` method, rendering the `_template`; and
        formatted as an OpenAI formatted chat i.e. a `ChatType`, assuming that there's only one turn,
        being from the user with the content being the rendered `_template`.
        """
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    source_language=self.source_language,
                    target_language=self.target_language,
                    unit=self.unit or random.choice(["sentence", "phrase", "passage"]),
                    difficulty=self.difficulty
                    or random.choice(["elementary school", "high school", "college"]),
                    high_score=self.high_score or random.choice(["4", "4.5", "5"]),
                    low_score=self.low_score or random.choice(["2.5", "3", "3.5"]),
                ).strip(),
            }
        ]  # type: ignore

    @property
    def keys(self) -> List[str]:
        """Contains the `keys` that will be parsed from the `LLM` output into a Python dict."""
        return ["S1", "S2", "S3"]
