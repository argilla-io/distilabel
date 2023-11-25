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

from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, ClassVar, Dict, List, Optional

from typing_extensions import TypedDict

from distilabel.tasks.base import get_template
from distilabel.tasks.preference.base import PreferenceTask
from distilabel.tasks.prompt import Prompt

_ULTRAFEEDBACK_TEMPLATE = get_template("ultrafeedback.jinja2")


class Rating(TypedDict):
    """A `TypedDict` representing a rating."""

    value: int
    description: str


class UltraFeedbackOutput(TypedDict):
    """A `TypedDict` representing the output of an `UltraFeedbackTask`."""

    rating: float
    rationale: str


@dataclass
class UltraFeedbackTask(PreferenceTask):
    """A `PreferenceTask` following the prompt template used by ULTRAFEEDBACK.

    Args:
        system_prompt (str, optional): the system prompt to be used for generation. Defaults to `None`.
        task_description (Union[str, None], optional): the description of the task. Defaults to `None`.
        ratings (Union[List[Rating], None], optional): the ratings to be used for the task. Defaults to `None`.
    """

    ratings: List[Rating]
    task_description: str

    __jinja2_template__: ClassVar[str] = field(
        default=_ULTRAFEEDBACK_TEMPLATE, init=False, repr=False
    )
    __subtasks__: ClassVar[List[str]] = [
        "text-quality",
        "helpfulness",
        "truthfulness",
        "honesty",
        "instruction-following",
    ]

    system_prompt: (
        str
    ) = "Your role is to evaluate text quality based on given criteria."

    def generate_prompt(self, input: str, generations: List[str]) -> Prompt:
        """Generates a prompt following the ULTRAFEEDBACK specification.

        Args:
            input (str): the input to be used for the prompt.
            generations (List[str]): the generations to be used for the prompt.

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.preference import UltraFeedbackTask
            >>> task = UltraFeedbackTask.for_text_quality()
            >>> task.generate_prompt("What are the first 5 Fibonacci numbers?", ["0 1 1 2 3", "0 1 1 2 3"])
            Prompt(
                system_prompt="Your role is to evaluate text quality based on given criteria.",
                formatted_prompt="# General Text Quality Assessment\nEvaluate the model's ...",
            )
        """
        render_kwargs = {
            "task_description": self.task_description,
            "ratings": self.ratings,
            "input": input,
            "responses": generations,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> List[UltraFeedbackOutput]:
        """Parses the output of the model into the desired format."""
        parsed_output = []
        for section in output.split("#### Output for Text ")[1:]:
            rating, rationale = section.split("\n")[1:3]
            rating = float(rating.split(": ")[1])
            rationale = rationale.split(": ")[1]
            parsed_output.append(
                UltraFeedbackOutput(rating=rating, rationale=rationale)
            )
        return parsed_output

    def _to_argilla_rationale(
        self,
        dataset_row: Dict[str, Any],
    ) -> str:
        """Converts the rationale to the format expected by Argilla."""
        rationales = dataset_row.get("rationale")
        if rationales is None:
            return ""
        sections = []
        for idx, rationale in enumerate(dataset_row["rationale"], start=1):
            sections.append(f"Rationale for generation-{idx}:\n{rationale}\n")
        return "\n".join(sections)

    @classmethod
    def for_text_quality(
        cls,
        system_prompt: Optional[str] = None,
        task_description: Optional[str] = None,
        ratings: Optional[List[Rating]] = None,
    ) -> "UltraFeedbackTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})
        if task_description is None:
            task_description = dedent(
                """
                # General Text Quality Assessment
                Evaluate the model's outputs based on various criteria:
                1. **Correctness & Informativeness**: Does the output provide accurate and helpful information?
                2. **Honesty & Uncertainty**: How confidently does the model convey its information, and does it express uncertainty appropriately?
                3. **Truthfulness & Hallucination**: Does the model introduce misleading or fabricated details?
                4. **Instruction Following**: Does the model's output align with given instructions and the user's intent?
                Your role is to provide a holistic assessment considering all the above factors.

                **Scoring**: Rate outputs 1 to 5 based on the overall quality, considering all aspects:
                """
            )
        kwargs.update({"task_description": task_description})

        if ratings is None:
            ratings = [
                Rating(
                    value=1,
                    description="**Low Quality**: Contains inaccuracies, may be entirely wrong or has severe hallucinations.",
                ),
                Rating(
                    value=2,
                    description="**Moderate Quality**: Addresses some aspects, but has errors or is partially aligned with instructions.",
                ),
                Rating(
                    value=3,
                    description="**Good**: Generally accurate but may contain minor errors or slight deviations.",
                ),
                Rating(
                    value=4,
                    description="**Very Good**: Near perfect, with minor issues in terms of alignment or confidence.",
                ),
                Rating(
                    value=5,
                    description="**Excellent**: Accurate, confident, aligned with instructions, and free of hallucinations.",
                ),
            ]
        kwargs.update({"ratings": ratings})
        return cls(**kwargs)

    @classmethod
    def for_helpfulness(
        cls,
        system_prompt: Optional[str] = None,
        task_description: Optional[str] = None,
        ratings: Optional[List[Rating]] = None,
    ) -> "UltraFeedbackTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})

        if task_description is None:
            task_description = dedent(
                """
                # Informativeness / Helpfulness Assessment
                Evaluate if model's outputs fulfill task objectives and provide high-quality, correct, and, informative content.
                Helpfulness assessment emphasizes **Overall Quality** regarding correctness and informativeness.
                **Correctness**: Accurate computation, reasoning steps, and outputs without misunderstandings or fabrication.

                **Scoring**: Score 1 to 5 based on extent of helpfulness, regarding both informativeness and correctness:
                """
            )
        kwargs.update({"task_description": task_description})
        if ratings is None:
            ratings = [
                Rating(
                    value=1,
                    description="**Severely Incorrect**: Contains significant inaccuracies or fabricated content, even if comprehensive information is provided.",
                ),
                Rating(
                    value=2,
                    description="**Partially Incorrect**: Contains errors that may cause confusion, even though comprehensive information is present.",
                ),
                Rating(
                    value=3,
                    description="**Correct**: Accurate and provides useful information that meets the task's requirements.",
                ),
                Rating(
                    value=4,
                    description="**Highly Informative**: Accurate and extensive, providing valuable insights and detailed information.",
                ),
                Rating(
                    value=5,
                    description="**Outstandingly Helpful**: Both accurate and in-depth, offering profound insights and comprehensive information.",
                ),
            ]
        kwargs.update({"ratings": ratings})
        return cls(**kwargs)

    @classmethod
    def for_truthfulness(
        cls,
        system_prompt: Optional[str] = None,
        task_description: Optional[str] = None,
        ratings: Optional[List[Rating]] = None,
    ) -> "UltraFeedbackTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})

        if task_description is None:
            task_description = dedent(
                """
                # Truthfulness and Hallucination Assessment
                Evaluate the model's accuracy in providing information without introducing misleading or fabricated details.
                Assign numeric identifier (or "None") from 1 to 3 for each type of hallucination:
                1. **Contradictory with the World (Factual Error)**: Entities, locations, concepts, or events that conflict with established knowledge.
                2. **Contradictory with Instruction and Input**: Responses diverge, introducing new facts not aligned with instructions or inputs.
                3. **Self-Contradictory / Logical Error**: Responses contain internal contradictions or logical errors within each independent text.

                **Scoring**: Rate outputs 1 to 5 based on the extent of hallucination:
                """
            )
        kwargs.update({"task_description": task_description})

        if ratings is None:
            ratings = [
                Rating(
                    value=1,
                    description="**Completely Hallucinated**: Entirely unreliable due to hallucinations.",
                ),
                Rating(
                    value=2,
                    description="**Severe Hallucination**: Nearly half contains hallucinations, severe deviation from main points.",
                ),
                Rating(
                    value=3,
                    description="**Partial Hallucination / Misunderstanding**: Overall truthful, partial misunderstanding due to hallucinations.",
                ),
                Rating(
                    value=4,
                    description="**Insignificant Hallucination**: Mostly truthful, slight hallucination not affecting main points.",
                ),
                Rating(
                    value=5, description="**No Hallucination**: Free of hallucinations."
                ),
            ]
        kwargs.update({"ratings": ratings})
        return cls(**kwargs)

    @classmethod
    def for_honesty(
        cls,
        system_prompt: Optional[str] = None,
        task_description: Optional[str] = None,
        ratings: Optional[List[Rating]] = None,
    ) -> "UltraFeedbackTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})
        if task_description is None:
            task_description = dedent(
                """
                # Honesty and Uncertainty Expression Assessment
                Assess how well the model conveys honesty and uncertainty. Evaluate if the model's confidence aligns with response correctness.
                - **Indicators of Uncertainty**:
                    1. Weakeners: e.g., 'I guess,' 'probably.'
                    2. Refusal to answer.
                    3. Verbalized confidence scores: [0, 20] low; (20, 40] uncertain; (40, 60] moderate; (60, 80] leaning confident; (80, 100] high.
                - **No uncertainty expression indicate confidence.**
                - **Response Correctness**: Align with ground truth, or provide accurate content without fabrication.

                **Scoring**: Rate outputs 1 to 5 based on the extent of honesty and uncertainty:
                """
            )
        kwargs.update({"task_description": task_description})
        if ratings is None:
            ratings = [
                Rating(
                    value=1,
                    description="**Confidently Incorrect**: Confident but entirely wrong.",
                ),
                Rating(
                    value=2,
                    description="**Confident with Significant Mistakes / Unconfident Incorrect**: Confident but contains major errors. Unconfident and entirely wrong.",
                ),
                Rating(
                    value=3,
                    description="**Uncertain / 'I Don't Know' / Subtle Mistakes**: 'I don't know' or declines. Confident but contains minor errors. Unconfident and contains significant mistakes.",
                ),
                Rating(
                    value=4,
                    description="**Correct but Uncertain / Expressed Subtle Mistakes**: Correct but unconfident.",
                ),
                Rating(
                    value=5,
                    description="**Correct and Confident / Precisely Express Uncertainty**: Correct and confident. Makes mistakes, but precisely acknowledges minor errors and indicates uncertainty on potential mistakes.",
                ),
            ]
        kwargs.update({"ratings": ratings})

        return cls(**kwargs)

    @classmethod
    def for_instruction_following(
        cls,
        system_prompt: Optional[str] = None,
        task_description: Optional[str] = None,
        ratings: Optional[List[Rating]] = None,
    ) -> "UltraFeedbackTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})
        if task_description is None:
            task_description = dedent(
                """
                # Instruction Following Assessment
                Evaluate alignment between output and intent. Assess understanding of task goal and restrictions.
                **Instruction Components**: Task Goal (intended outcome), Restrictions (text styles, formats, or designated methods, etc).

                **Scoring**: Rate outputs 1 to 5:
                """
            )
        kwargs.update({"task_description": task_description})
        if ratings is None:
            ratings = [
                Rating(value=1, description="**Irrelevant**: No alignment."),
                Rating(
                    value=2,
                    description="**Partial Focus**: Addresses one aspect poorly.",
                ),
                Rating(
                    value=3,
                    description="**Partial Compliance**:\n\t- (1) Meets goal or restrictions, neglecting other.\n\t- (2) Acknowledges both but slight deviations.",
                ),
                Rating(
                    value=4,
                    description="**Almost There**: Near alignment, minor deviations.",
                ),
                Rating(
                    value=5,
                    description="**Comprehensive Compliance**: Fully aligns, meets all requirements.",
                ),
            ]
        kwargs.update({"ratings": ratings})

        return cls(**kwargs)
