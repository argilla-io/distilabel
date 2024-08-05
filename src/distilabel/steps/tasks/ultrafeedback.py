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

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from typing import Any, Dict, List, Literal, Optional, Union

from jinja2 import Template
from pydantic import PrivateAttr

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

    Input columns:
        - instruction (`str`): The reference instruction to evaluate the text outputs.
        - generations (`List[str]`): The text outputs to evaluate for the given instruction.

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
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        ultrafeedback = UltraFeedback(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            )
        )

        ultrafeedback.load()

        result = next(
            chat.process(
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

    _system_prompt: str = PrivateAttr(
        default=(
            "Your role is to evaluate text quality based on given criteria.\n"
            'You\'ll receive an instructional description ("Instruction") and {no_texts} text outputs ("Text").\n'
            "Understand and interpret instructions to evaluate effectively.\n"
            "Provide annotations for each text with a rating and rationale.\n"
            "The {no_texts} texts given are independent, and should be evaluated separately.\n"
        )
    )
    _template: Optional["Template"] = PrivateAttr(default=...)

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
        return ["instruction", "generations"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        return [
            {
                "role": "system",
                "content": self._system_prompt.format(
                    no_texts=len(input["generations"])
                ),
            },
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    instruction=input["instruction"], generations=input["generations"]
                ),
            },
        ]

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
        self, output: Union[str, None], input: Dict[str, Any]
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
                    "ratings": int(re.findall(r"\b\d+\b", matches.group(1))[0])
                    if matches.group(1) not in ["None", "N/A"]
                    else None,
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
                    "types": int(re.findall(r"\b\d+\b", matches.group(1))[0])
                    if matches.group(1) not in ["None", "N/A"]
                    else None,
                    "rationales": matches.group(2),
                    "ratings": int(re.findall(r"\b\d+\b", matches.group(3))[0])
                    if matches.group(3) not in ["None", "N/A"]
                    else None,
                    "rationales-for-ratings": matches.group(4),
                }
            )
        return group_dicts(*formatted_outputs)
