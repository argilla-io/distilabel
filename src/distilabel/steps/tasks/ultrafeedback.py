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

import importlib.resources as importlib_resources
import re
from typing import Any, Dict, List, Literal, Optional, Union

import orjson
from jinja2 import Template
from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.utils.dicts import group_dicts


class UltraFeedback(Task):
    """Rank generations focusing on different aspects using an `LLM`.

    UltraFeedback: Boosting Language Models with High-quality Feedback.

    Attributes:
        aspect: The aspect to perform with the `UltraFeedback` model. The available aspects are:
            - `helpfulness`: Evaluate text outputs based on helpfulness.
            - `honesty`: Evaluate text outputs based on honesty.
            - `instruction-following`: Evaluate text outputs based on given instructions.
            - `truthfulness`: Evaluate text outputs based on truthfulness.
            Additionally, a custom aspect has been defined by Argilla, so as to evaluate the overall
            assessment of the text outputs within a single prompt. The custom aspect is:
            - `overall-rating`: Evaluate text outputs based on an overall assessment.
            Defaults to `"overall-rating"`.
        system_prompt: The system prompt to use in the generation. If not provided, then
            it will check if the input row has a column named `system_prompt` and use it.
            If not, then no system prompt will be used. Defaults to `None`.

    Input columns:
        - instruction (`str`): The reference instruction to evaluate the text outputs.
        - generations (`List[str]`): The text outputs to evaluate for the given instruction.
        - system_prompt (`str`): The system prompt to use in the generation.

    Output columns:
        - ratings (`List[float]`): The ratings for each of the provided text outputs.
        - rationales (`List[str]`): The rationales for each of the provided text outputs.
        - model_name (`str`): The name of the model used to generate the ratings and rationales.

    Categories:
        - preference

    References:
        - [`UltraFeedback: Boosting Language Models with High-quality Feedback`](https://arxiv.org/abs/2310.01377)
        - [`UltraFeedback - GitHub Repository`](https://github.com/OpenBMB/UltraFeedback)

    Examples:
        Rate generations from different LLMs based on the selected aspect:

        ```python
        from distilabel.steps.tasks import UltraFeedback
        from distilabel.models import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        ultrafeedback = UltraFeedback(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            ),
            use_default_structured_output=False
        )

        ultrafeedback.load()

        result = next(
            ultrafeedback.process(
                [
                    {
                        "instruction": "How much is 2+2?",
                        "generations": ["4", "and a car"],
                    }
                ]
            )
        )
        # result
        # [
        #     {
        #         'instruction': 'How much is 2+2?',
        #         'generations': ['4', 'and a car'],
        #         'ratings': [1, 2],
        #         'rationales': ['explanation for 4', 'explanation for and a car'],
        #         'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
        #     }
        # ]
        ```

        Rate generations from different LLMs based on the honesty, using the default structured output:

        ```python
        from distilabel.steps.tasks import UltraFeedback
        from distilabel.models import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        ultrafeedback = UltraFeedback(
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            aspect="honesty"
        )

        ultrafeedback.load()

        result = next(
            ultrafeedback.process(
                [
                    {
                        "instruction": "How much is 2+2?",
                        "generations": ["4", "and a car"],
                    }
                ]
            )
        )
        # result
        # [{'instruction': 'How much is 2+2?',
        # 'generations': ['4', 'and a car'],
        # 'ratings': [5, 1],
        # 'rationales': ['The response is correct and confident, as it directly answers the question without expressing any uncertainty or doubt.',
        # "The response is confidently incorrect, as it provides unrelated information ('a car') and does not address the question. The model shows no uncertainty or indication that it does not know the answer."],
        # 'distilabel_metadata': {'raw_output_ultra_feedback_0': '{"ratings": [\\n    5,\\n    1\\n] \\n\\n,"rationales": [\\n    "The response is correct and confident, as it directly answers the question without expressing any uncertainty or doubt.",\\n    "The response is confidently incorrect, as it provides unrelated information (\'a car\') and does not address the question. The model shows no uncertainty or indication that it does not know the answer."\\n] }'},
        # 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
        ```

        Rate generations from different LLMs based on the helpfulness, using the default structured output:

        ```python
        from distilabel.steps.tasks import UltraFeedback
        from distilabel.models import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        ultrafeedback = UltraFeedback(
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                generation_kwargs={"max_new_tokens": 512},
            ),
            aspect="helpfulness"
        )

        ultrafeedback.load()

        result = next(
            ultrafeedback.process(
                [
                    {
                        "instruction": "How much is 2+2?",
                        "generations": ["4", "and a car"],
                    }
                ]
            )
        )
        # result
        # [{'instruction': 'How much is 2+2?',
        #   'generations': ['4', 'and a car'],
        #   'ratings': [1, 5],
        #   'rationales': ['Text 1 is clear and relevant, providing the correct answer to the question. It is also not lengthy and does not contain repetition. However, it lacks comprehensive information or detailed description.',
        #    'Text 2 is neither clear nor relevant to the task. It does not provide any useful information and seems unrelated to the question.'],
        #   'rationales_for_rating': ['Text 1 is rated as Correct (3) because it provides the accurate answer to the question, but lacks comprehensive information or detailed description.',
        #    'Text 2 is rated as Severely Incorrect (1) because it does not provide any relevant information and seems unrelated to the question.'],
        #   'types': [1, 3, 1],
        #   'distilabel_metadata': {'raw_output_ultra_feedback_0': '{ \\n  "ratings": [\\n    1,\\n    5\\n  ]\\n ,\\n  "rationales": [\\n    "Text 1 is clear and relevant, providing the correct answer to the question. It is also not lengthy and does not contain repetition. However, it lacks comprehensive information or detailed description.",\\n    "Text 2 is neither clear nor relevant to the task. It does not provide any useful information and seems unrelated to the question."\\n  ]\\n ,\\n  "rationales_for_rating": [\\n    "Text 1 is rated as Correct (3) because it provides the accurate answer to the question, but lacks comprehensive information or detailed description.",\\n    "Text 2 is rated as Severely Incorrect (1) because it does not provide any relevant information and seems unrelated to the question."\\n  ]\\n ,\\n  "types": [\\n    1, 3,\\n    1\\n  ]\\n  }'},
        #   'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
        ```

    Citations:
        ```
        @misc{cui2024ultrafeedbackboostinglanguagemodels,
            title={UltraFeedback: Boosting Language Models with Scaled AI Feedback},
            author={Ganqu Cui and Lifan Yuan and Ning Ding and Guanming Yao and Bingxiang He and Wei Zhu and Yuan Ni and Guotong Xie and Ruobing Xie and Yankai Lin and Zhiyuan Liu and Maosong Sun},
            year={2024},
            eprint={2310.01377},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2310.01377},
        }
        ```
    """

    aspect: Literal[
        "helpfulness",
        "honesty",
        "instruction-following",
        "truthfulness",
        # Custom aspects
        "overall-rating",
    ] = "overall-rating"

    system_prompt: str = PrivateAttr(
        default=(
            "Your role is to evaluate text quality based on given criteria.\n"
            'You\'ll receive an instructional description ("Instruction") and {no_texts} text outputs ("Text").\n'
            "Understand and interpret instructions to evaluate effectively.\n"
            "Provide annotations for each text with a rating and rationale.\n"
            "The {no_texts} texts given are independent, and should be evaluated separately.\n"
        )
    )
    _template: Optional["Template"] = PrivateAttr(default=...)
    _can_be_used_with_offline_batch_generation = True

    def load(self) -> None:
        """Loads the Jinja2 template for the given `aspect`."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "ultrafeedback"
            / f"{self.aspect}.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`, and the `generations` for it."""
        return {"instruction": True, "generations": True, "system_prompt": False}

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        messages = []
        if "system_prompt" in input:
            messages.append({"role": "system", "content": input["system_prompt"]})
        elif self.system_prompt:
            messages.append(
                {
                    "role": "user",
                    "content": self.system_prompt.format(
                        no_texts=len(input["generations"])
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    instruction=input["instruction"], generations=input["generations"]
                ),
            },
        )
        return messages

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        columns = []
        if self.aspect in ["honesty", "instruction-following", "overall-rating"]:
            columns = ["ratings", "rationales"]
        elif self.aspect in ["helpfulness", "truthfulness"]:
            columns = ["types", "rationales", "ratings", "rationales-for-ratings"]
        return columns + ["model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `ratings` and `rationales` for
        each of the provided `generations` for the given `instruction`. The `model_name`
        will be automatically included within the `process` method of `Task`.

        Args:
            output: a string representing the output of the LLM via the `process` method.
            input: the input to the task, as required by some tasks to format the output.

        Returns:
            A dictionary containing either the `ratings` and `rationales` for each of the provided
            `generations` for the given `instruction` if the provided aspect is either `honesty`,
            `instruction-following`, or `overall-rating`; or the `types`, `rationales`,
            `ratings`, and `rationales-for-ratings` for each of the provided `generations` for the
            given `instruction` if the provided aspect is either `helpfulness` or `truthfulness`.
        """
        assert input is not None, "Input is required to format the output."

        if self.aspect in [
            "honesty",
            "instruction-following",
            "overall-rating",
        ]:
            return self._format_ratings_rationales_output(output, input)

        return self._format_types_ratings_rationales_output(output, input)

    def _format_ratings_rationales_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, List[Any]]:
        """Formats the output when the aspect is either `honesty`, `instruction-following`, or `overall-rating`."""
        if output is None:
            return {
                "ratings": [None] * len(input["generations"]),
                "rationales": [None] * len(input["generations"]),
            }

        if self.use_default_structured_output:
            return self._format_structured_output(output, input)

        pattern = r"Rating: (.+?)\nRationale: (.+)"
        sections = output.split("\n\n")

        formatted_outputs = []
        for section in sections:
            matches = None
            if section is not None and section != "":
                matches = re.search(pattern, section, re.DOTALL)
            if not matches:
                formatted_outputs.append({"ratings": None, "rationales": None})
                continue

            formatted_outputs.append(
                {
                    "ratings": (
                        int(re.findall(r"\b\d+\b", matches.group(1))[0])
                        if matches.group(1) not in ["None", "N/A"]
                        else None
                    ),
                    "rationales": matches.group(2),
                }
            )
        return group_dicts(*formatted_outputs)

    def _format_types_ratings_rationales_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, List[Any]]:
        """Formats the output when the aspect is either `helpfulness` or `truthfulness`."""
        if output is None:
            return {
                "types": [None] * len(input["generations"]),
                "rationales": [None] * len(input["generations"]),
                "ratings": [None] * len(input["generations"]),
                "rationales-for-ratings": [None] * len(input["generations"]),
            }

        if self.use_default_structured_output:
            return self._format_structured_output(output, input)

        pattern = r"Type: (.+?)\nRationale: (.+?)\nRating: (.+?)\nRationale: (.+)"

        sections = output.split("\n\n")

        formatted_outputs = []
        for section in sections:
            matches = None
            if section is not None and section != "":
                matches = re.search(pattern, section, re.DOTALL)
            if not matches:
                formatted_outputs.append(
                    {
                        "types": None,
                        "rationales": None,
                        "ratings": None,
                        "rationales-for-ratings": None,
                    }
                )
                continue

            formatted_outputs.append(
                {
                    "types": (
                        int(re.findall(r"\b\d+\b", matches.group(1))[0])
                        if matches.group(1) not in ["None", "N/A"]
                        else None
                    ),
                    "rationales": matches.group(2),
                    "ratings": (
                        int(re.findall(r"\b\d+\b", matches.group(3))[0])
                        if matches.group(3) not in ["None", "N/A"]
                        else None
                    ),
                    "rationales-for-ratings": matches.group(4),
                }
            )
        return group_dicts(*formatted_outputs)

    @override
    def get_structured_output(self) -> Dict[str, Any]:
        """Creates the json schema to be passed to the LLM, to enforce generating
        a dictionary with the output which can be directly parsed as a python dictionary.

        The schema corresponds to the following:

        ```python
        from pydantic import BaseModel
        from typing import List

        class SchemaUltraFeedback(BaseModel):
            ratings: List[int]
            rationales: List[str]

        class SchemaUltraFeedbackWithType(BaseModel):
            types: List[Optional[int]]
            ratings: List[int]
            rationales: List[str]
            rationales_for_rating: List[str]
        ```

        Returns:
            JSON Schema of the response to enforce.
        """
        if self.aspect in [
            "honesty",
            "instruction-following",
            "overall-rating",
        ]:
            return {
                "properties": {
                    "ratings": {
                        "items": {"type": "integer"},
                        "title": "Ratings",
                        "type": "array",
                    },
                    "rationales": {
                        "items": {"type": "string"},
                        "title": "Rationales",
                        "type": "array",
                    },
                },
                "required": ["ratings", "rationales"],
                "title": "SchemaUltraFeedback",
                "type": "object",
            }
        return {
            "properties": {
                "types": {
                    "items": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "title": "Types",
                    "type": "array",
                },
                "ratings": {
                    "items": {"type": "integer"},
                    "title": "Ratings",
                    "type": "array",
                },
                "rationales": {
                    "items": {"type": "string"},
                    "title": "Rationales",
                    "type": "array",
                },
                "rationales_for_rating": {
                    "items": {"type": "string"},
                    "title": "Rationales For Rating",
                    "type": "array",
                },
            },
            "required": ["types", "ratings", "rationales", "rationales_for_rating"],
            "title": "SchemaUltraFeedbackWithType",
            "type": "object",
        }

    def _format_structured_output(
        self, output: str, input: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            if self.aspect in [
                "honesty",
                "instruction-following",
                "overall-rating",
            ]:
                return {
                    "ratings": [None] * len(input["generations"]),
                    "rationales": [None] * len(input["generations"]),
                }
            return {
                "ratings": [None] * len(input["generations"]),
                "rationales": [None] * len(input["generations"]),
                "types": [None] * len(input["generations"]),
                "rationales-for-ratings": [None] * len(input["generations"]),
            }

    @override
    def _sample_input(self) -> ChatType:
        return self.format_input(
            {
                "instruction": f"<PLACEHOLDER_{'instruction'.upper()}>",
                "generations": [
                    f"<PLACEHOLDER_{f'GENERATION_{i}'.upper()}>" for i in range(2)
                ],
            }
        )
