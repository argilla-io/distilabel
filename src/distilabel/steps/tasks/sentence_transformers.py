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
import sys
from typing import TYPE_CHECKING, Any, Dict, Final, List, Literal, Optional, Union

from jinja2 import Template

from distilabel.steps.tasks.base import Task

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType

GenerationAction = Literal["paraphrase", "semantically-similar", "query"]

POSITIVE_NEGATIVE_PAIR_REGEX = re.compile(
    r"## Positive\s+(.*?)\s+(?:## Negative\s+(.*?)\s*)?$",
    re.DOTALL,
)


GENERATION_ACTION_SENTENCES: Final[Dict[GenerationAction, str]] = {
    "paraphrase": "paraphrase",
    "semantically-similar": "be semantically similar to",
    "query": "be a query for",
}


class GenerateSentencePair(Task):
    """Generate a positive and negative (optionally) sentences given an anchor sentence.

    `GenerateSentencePair` is a pre-defined task that given an anchor sentence generates
    a positive sentence related to the anchor and optionally a negative sentence unrelated
    to the anchor. This task is useful to generate training datasets for training embeddings
    models.

    Attributes:
        triplet: a flag to indicate if the task should generate a triplet of sentences
            (anchor, positive, negative). Defaults to `False`.
        action: the action to perform to generate the positive sentence.

    Input columns:
        - anchor (`str`): The anchor sentence to generate the positive and negative sentences.

    Output columns:
        - positive (`str`): The positive sentence related to the `anchor`.
        - negative (`str`): The negative sentence unrelated to the `anchor`.

    Categories:
        - embedding
    """

    triplet: bool = False
    action: GenerationAction

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "generate-sentence-pair.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        return ["anchor"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {
                "role": "user",
                "content": self._template.render(
                    anchor=input["anchor"],
                    action=GENERATION_ACTION_SENTENCES[self.action],
                ),
            }
        ]

    @property
    def outputs(self) -> List[str]:
        if self.triplet:
            return ["positive", "negative"]

        return ["positive"]

    def format_output(
        self, output: Union[str, None], input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if output is None:
            return {"positive": None, "negative": None}

        match = POSITIVE_NEGATIVE_PAIR_REGEX.match(output)
        if match is None:
            return {"positive": None, "negative": None}

        groups = match.groups()
        if self.triplet:
            return {
                "positive": groups[0].strip(),
                "negative": groups[1].strip() if len(groups) > 1 else None,
            }

        return {"positive": groups[0].strip()}
