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
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks.base import GeneratorTask, Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import GeneratorStepOutput


class TextRetrievalTaskGeneration(GeneratorTask):
    """Generate text retrieval task descriptions using an `LLM`.

    ...

    References:
        - [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)
    """

    flatten: bool = False

    _prompt: ChatType = PrivateAttr(
        default=[
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
                    "Your output should always be a python list of strings only, with about 20 elements, and each element\n"
                    "corresponds to a distinct retrieval task in one sentence. Do not explain yourself or output anything else. Be\n"
                    "creative!"
                ),
            },
        ]
    )

    @override
    def process(self, offset: int = 0) -> GeneratorStepOutput:  # type: ignore
        formatted_inputs = [self._prompt]
        outputs = self.llm.generate(
            inputs=formatted_inputs,
            num_generations=self.num_generations,
            **self.llm.generation_kwargs,  # type: ignore
        )

        task_outputs = []
        for input_outputs in outputs:
            formatted_outputs = self._format_outputs(input_outputs)  # type: ignore
            for formatted_output in formatted_outputs:
                if isinstance(formatted_output["tasks"], list) and self.flatten:
                    task_outputs.extend(
                        [
                            {"task": output, "model_name": self.llm.model_name}
                            for output in formatted_output["tasks"]
                        ]
                    )
                else:
                    task_outputs.append(
                        {
                            "task" if not self.flatten else "tasks": formatted_output[
                                "tasks"
                            ],
                            "model_name": self.llm.model_name,
                        }
                    )
        yield task_outputs, True

    @property
    def outputs(self) -> List[str]:
        return ["tasks" if not self.flatten else "task", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        try:
            if output is not None:
                output = eval(output)
        except Exception:
            pass
        return {"tasks": output}


class TextRetrievalGeneration(Task):
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

    seed: int = 42

    def load(self) -> None:
        super().load()

        random.seed(self.seed)

    @property
    def inputs(self) -> List[str]:
        return ["task"]

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
                    f"- The 'user_query' should be {self.query_type}, {self.query_length}, {self.clarity}, and diverse in topic.\n"
                    "- All documents must be created independent of the query. Avoid copying the query verbatim. It’s acceptable\n"
                    "if some parts of the 'positive_document' are not topically related to the query.\n"
                    f"- All documents should be at least {self.num_words} words long.\n"
                    "- The 'hard_negative_document' contains some useful information, but it should be less useful or comprehensive compared to the 'positive_document'.\n"
                    f"- Both the query and documents should be in {self.language}.\n"
                    "- Do not provide any explanation in any document on why it is relevant or not relevant to the query.\n"
                    f"- Both the query and documents require {self.difficulty} level education to understand.\n"
                    "Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"
                ).replace("'", '"'),
            },
        ]

    @property
    def outputs(self) -> List[str]:
        return [
            "user_query",
            "positive_document",
            "hard_negative_document",
            "model_name",
        ]

    def format_output(
        self, output: Union[str, None], input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if output is None:
            return {
                "user_query": None,
                "positive_document": None,
                "hard_negative_document": None,
            }

        try:
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
            return {
                "user_query": None,
                "positive_document": None,
                "hard_negative_document": None,
            }

        return {
            "user_query": output.get("user_query", None),
            "positive_document": output.get("positive_document", None),
            "hard_negative_document": output.get("hard_negative_document", None),
        }
