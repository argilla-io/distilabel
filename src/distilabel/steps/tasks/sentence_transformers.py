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

import orjson
from jinja2 import Template
from typing_extensions import override

from distilabel.steps.tasks.base import Task

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType

GenerationAction = Literal["paraphrase", "semantically-similar", "query", "answer"]

POSITIVE_NEGATIVE_PAIR_REGEX = re.compile(
    r"## Positive\s+(.*?)(?:\s+## Negative\s+(.*?))?\s*$",
    re.DOTALL,
)

GENERATION_ACTION_SENTENCES: Final[Dict[GenerationAction, str]] = {
    "paraphrase": "paraphrase",
    "semantically-similar": "be semantically similar to",
    "query": "be a query for",
    "answer": "be an answer for",
}

POSITIVE_SYSTEM_PROMPT: str = (
    "Your task is to generate a positive sentence given an anchor sentence.{context} The positive"
    " sentence has to {action_sentence} the anchor sentence. You must output only one new"
    " section: `## Positive`."
)

NEGATIVE_STYLE: Dict[str, str] = {
    "negative": "can use similar words but must not be related to the anchor sentence",
    "hard-negative": (
        "is a 'hard negative' that meets the following criteria:\n"
        "- Uses similar keywords or phrases as the anchor sentence\n"
        "- Has a similar grammatical structure or syntax\n"
        "- Is not related to the anchor sentence, but could be mistaken for it\n"
        "Try to create a negative sentence that would be challenging for a model to distinguish "
        "from the positive sentence"
    ),
}

POSITIVE_NEGATIVE_SYSTEM_PROMPT: str = (
    "Your task is to generate a positive and a negative sentence given an anchor sentence.{context}"
    " The positive sentence has to {action_sentence} the anchor sentence, while the negative"
    " sentence {negative_style}. You must output only two new sections: `## Positive` and `## Negative`."
)

CONTEXT_INTRO: Final[str] = " Take into account the context given."


class GenerateSentencePair(Task):
    """Generate a positive and negative (optionally) sentences given an anchor sentence.

    `GenerateSentencePair` is a pre-defined task that given an anchor sentence generates
    a positive sentence related to the anchor and optionally a negative sentence unrelated
    to the anchor or similar to it. Optionally, you can give a context to guide the LLM
    towards more specific behavior. This task is useful to generate training datasets for
    training embeddings models.

    Attributes:
        triplet: a flag to indicate if the task should generate a triplet of sentences
            (anchor, positive, negative). Defaults to `False`.
        action: the action to perform to generate the positive sentence.
        context: the context to use for the generation. Can be helpful to guide the LLM
            towards more specific context. Not used by default.
        hard_negative: A flag to indicate if the negative should be a hard-negative or not.
            Hard negatives make it hard for the model to distinguish against the positive,
            with a higher degree of semantic similarity.

    Input columns:
        - anchor (`str`): The anchor sentence to generate the positive and negative sentences.

    Output columns:
        - positive (`str`): The positive sentence related to the `anchor`.
        - negative (`str`): The negative sentence unrelated to the `anchor` if `triplet=True`,
            or more similar to the positive to make it more challenging for a model to distinguish
            in case `hard_negative=True`.
        - model_name (`str`): The name of the model that was used to generate the sentences.

    Categories:
        - embedding

    Examples:
        Paraphrasing:

        ```python
        from distilabel.steps.tasks import GenerateSentencePair
        from distilabel.llms import InferenceEndpointsLLM

        generate_sentence_pair = GenerateSentencePair(
            triplet=True, # `False` to generate only positive
            action="paraphrase",
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            input_batch_size=10,
        )

        generate_sentence_pair.load()

        result = generate_sentence_pair.process([{"anchor": "What Game of Thrones villain would be the most likely to give you mercy?"}])
        ```

        Generating semantically similar sentences:

        ```python
        from distilabel.llms import InferenceEndpointsLLM
        from distilabel.steps.tasks import GenerateSentencePair

        generate_sentence_pair = GenerateSentencePair(
            triplet=True, # `False` to generate only positive
            action="semantically-similar",
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            input_batch_size=10,
        )

        generate_sentence_pair.load()

        result = generate_sentence_pair.process([{"anchor": "How does 3D printing work?"}])
        ```

        Generating queries:

        ```python
        from distilabel.steps.tasks import GenerateSentencePair
        from distilabel.llms import InferenceEndpointsLLM

        generate_sentence_pair = GenerateSentencePair(
            triplet=True, # `False` to generate only positive
            action="query",
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            input_batch_size=10,
        )

        generate_sentence_pair.load()

        result = generate_sentence_pair.process([{"anchor": "Argilla is an open-source data curation platform for LLMs. Using Argilla, ..."}])
        ```

        Generating answers:

        ```python
        from distilabel.steps.tasks import GenerateSentencePair
        from distilabel.llms import InferenceEndpointsLLM

        generate_sentence_pair = GenerateSentencePair(
            triplet=True, # `False` to generate only positive
            action="answer",
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            input_batch_size=10,
        )

        generate_sentence_pair.load()

        result = generate_sentence_pair.process([{"anchor": "What Game of Thrones villain would be the most likely to give you mercy?"}])
        ```

        Generating queries with context (**applies to every action**):

        ```python
        from distilabel.steps.tasks import GenerateSentencePair
        from distilabel.llms import InferenceEndpointsLLM

        generate_sentence_pair = GenerateSentencePair(
            triplet=True, # `False` to generate only positive
            action="query",
            context="Argilla is an open-source data curation platform for LLMs.",
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            input_batch_size=10,
        )

        generate_sentence_pair.load()

        result = generate_sentence_pair.process([{"anchor": "I want to generate queries for my LLM."}])
        ```

        Generating Hard-negatives (**applies to every action**):

        ```python
        from distilabel.steps.tasks import GenerateSentencePair
        from distilabel.llms import InferenceEndpointsLLM

        generate_sentence_pair = GenerateSentencePair(
            triplet=True, # `False` to generate only positive
            action="query",
            context="Argilla is an open-source data curation platform for LLMs.",
            hard_negative=True,
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            input_batch_size=10,
        )

        generate_sentence_pair.load()

        result = generate_sentence_pair.process([{"anchor": "I want to generate queries for my LLM."}])
        ```

        Generating structured data with default schema (**applies to every action**):

        ```python
        from distilabel.steps.tasks import GenerateSentencePair
        from distilabel.llms import InferenceEndpointsLLM

        generate_sentence_pair = GenerateSentencePair(
            triplet=True, # `False` to generate only positive
            action="query",
            context="Argilla is an open-source data curation platform for LLMs.",
            hard_negative=True,
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            input_batch_size=10,
            use_default_structured_output=True
        )

        generate_sentence_pair.load()

        result = generate_sentence_pair.process([{"anchor": "I want to generate queries for my LLM."}])
        ```
    """

    triplet: bool = False
    action: GenerationAction
    hard_negative: bool = False
    context: str = ""

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
        """The inputs for the task is the `anchor` sentence."""
        return ["anchor"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The inputs are formatted as a `ChatType`, with a system prompt describing the
        task of generating a positive and negative sentences for the anchor sentence. The
        anchor is provided as the first user interaction in the conversation.

        Args:
            input: The input containing the `anchor` sentence.

        Returns:
            A list of dictionaries containing the system and user interactions.
        """
        action_sentence = GENERATION_ACTION_SENTENCES[self.action]

        format_system_prompt = {
            "action_sentence": action_sentence,
            "context": CONTEXT_INTRO if self.context else "",
        }
        if self.triplet:
            format_system_prompt["negative_style"] = NEGATIVE_STYLE[
                "hard-negative" if self.hard_negative else "negative"
            ]

        system_prompt = (
            POSITIVE_NEGATIVE_SYSTEM_PROMPT if self.triplet else POSITIVE_SYSTEM_PROMPT
        ).format(**format_system_prompt)

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": self._template.render(
                    anchor=input["anchor"],
                    context=self.context if self.context else None,
                ),
            },
        ]

    @property
    def outputs(self) -> List[str]:
        """The outputs for the task are the `positive` and `negative` sentences, as well
        as the `model_name` used to generate the sentences."""
        columns = ["positive", "negative"] if self.triplet else ["positive"]
        columns += ["model_name"]
        return columns

    def format_output(
        self, output: Union[str, None], input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formats the output of the LLM, to extract the `positive` and `negative` sentences
        generated. If the output is `None` or the regex doesn't match, then the outputs
        will be set to `None` as well.

        Args:
            output: The output of the LLM.
            input: The input used to generate the output.

        Returns:
            The formatted output containing the `positive` and `negative` sentences.
        """
        if output is None:
            return {"positive": None, "negative": None}

        if self.use_default_structured_output:
            return self._format_structured_output(output)

        match = POSITIVE_NEGATIVE_PAIR_REGEX.match(output)
        if match is None:
            formatted_output = {"positive": None}
            if self.triplet:
                formatted_output["negative"] = None
            return formatted_output

        groups = match.groups()
        if self.triplet:
            return {
                "positive": groups[0].strip(),
                "negative": (
                    groups[1].strip()
                    if len(groups) > 1 and groups[1] is not None
                    else None
                ),
            }

        return {"positive": groups[0].strip()}

    @override
    def get_structured_output(self) -> Dict[str, Any]:
        """Creates the json schema to be passed to the LLM, to enforce generating
        a dictionary with the output which can be directly parsed as a python dictionary.

        Returns:
            JSON Schema of the response to enforce.
        """
        if self.triplet:
            return {
                "properties": {
                    "positive": {"title": "Positive", "type": "string"},
                    "negative": {"title": "Negative", "type": "string"},
                },
                "required": ["positive", "negative"],
                "title": "Schema",
                "type": "object",
            }
        return {
            "properties": {"positive": {"title": "Positive", "type": "string"}},
            "required": ["positive"],
            "title": "Schema",
            "type": "object",
        }

    def _format_structured_output(self, output: str) -> Dict[str, str]:
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
            if self.triplet:
                return {"positive": None, "negative": None}
            return {"positive": None}
