from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from typing_extensions import TypedDict

from ultralabel.tasks.base import Task, get_template
from ultralabel.tasks.utils import Prompt

try:
    import argilla as rg

    _argilla_installed = True
except ImportError:
    _argilla_installed = False

if TYPE_CHECKING:
    from argilla.client.feedback.schemas.records import FeedbackRecord
    from argilla.client.feedback.schemas.types import (
        AllowedFieldTypes,
        AllowedQuestionTypes,
    )

_ULTRAFEEDBACK_TEMPLATE = get_template("ultrafeedback.jinja2")


class Rating(TypedDict):
    value: int
    description: str


class UltraFeedbackOutput(TypedDict):
    rating: int
    rationale: str


class UltraFeedbackTask(Task):
    ratings: List[Rating]

    __jinja2_template__: str = _ULTRAFEEDBACK_TEMPLATE
    __subtasks__: List[str] = [
        "text-quality",
        "helpfulness",
        "truthfulness",
        "honesty",
        "instruction-following",
    ]

    system_prompt: str = (
        "Your role is to evaluate text quality based on given criteria."
    )

    @property
    def input_args_names(self) -> List[str]:
        return ["instruction", "generations"]

    @property
    def output_args_names(self) -> List[str]:
        return ["rating", "rationale"]

    def generate_prompt(self, instruction: str, generations: List[str]) -> Prompt:
        render_kwargs = {
            "task_description": self.task_description,
            "ratings": self.ratings,
            "instruction": instruction,
            "responses": generations,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> List[UltraFeedbackOutput]:
        parsed_output = []
        for section in output.split("#### Output for Text ")[1:]:
            rating, rationale = section.split("\n")[1:3]
            rating = int(rating.split(": ")[1])
            rationale = rationale.split(": ")[1]
            parsed_output.append(
                UltraFeedbackOutput(rating=rating, rationale=rationale)
            )
        return parsed_output

    def to_argilla_fields(
        self, dataset_row: Dict[str, Any]
    ) -> List["AllowedFieldTypes"]:
        if not _argilla_installed:
            raise ImportError("The argilla library is not installed.")
        argilla_fields = []
        for arg_name in self.input_args_names:
            if arg_name not in dataset_row:
                raise ValueError(
                    f"Dataset row does not contain the required field '{arg_name}'."
                )
            if isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    argilla_fields.append(rg.TextField(name=f"{arg_name}-{idx}"))
            elif isinstance(dataset_row[arg_name], str):
                argilla_fields.append(rg.TextField(name=arg_name))
            else:
                raise ValueError(
                    f"Type {type(dataset_row[arg_name])} is not supported."
                )
        return argilla_fields

    def to_argilla_questions(
        self,
        dataset_row: Dict[str, Any],
        group_ratings_as_ranking: bool = False,
    ) -> List["AllowedQuestionTypes"]:
        if not _argilla_installed:
            raise ImportError("The argilla library is not installed.")
        argilla_questions = []
        for arg_name in ["generations"]:
            if arg_name not in dataset_row:
                raise ValueError(
                    f"Dataset row does not contain the required field '{arg_name}'."
                )
            if isinstance(dataset_row[arg_name], list):
                # If `group_ratings_as_ranking` is True, then we group all the ratings into a ranking
                if group_ratings_as_ranking:
                    argilla_questions.append(
                        rg.RankingQuestion(
                            name=f"{arg_name}-ranking",
                            title=f"Rank the {arg_name} from best to worst.",
                            values=[
                                f"{arg_name}-{idx}"
                                for idx in range(1, len(dataset_row[arg_name]) + 1)
                            ],
                        )
                    )
                # Otherwise, we ask for each rating individually, but we still add the rationale
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    if not group_ratings_as_ranking:
                        argilla_questions.append(
                            rg.RatingQuestion(
                                name=f"{arg_name}-{idx}-rating",
                                title=f"What's the rating for {arg_name}-{idx}?",
                                values=list(range(1, len(self.ratings) + 1)),
                            ),
                        )
                    argilla_questions.append(
                        rg.TextQuestion(
                            name=f"{arg_name}-{idx}-rationale",
                            title=f"What's the rationale behind the rating for {arg_name}-{idx}?",
                        ),
                    )
        return argilla_questions

    def to_argilla_record(  # noqa: C901
        self, dataset_row: Dict[str, Any], group_ratings_as_ranking: bool = False
    ) -> "FeedbackRecord":
        if not _argilla_installed:
            raise ImportError("The argilla library is not installed.")
        fields = {}
        for input_arg_name in self.input_args_names:
            if isinstance(dataset_row[input_arg_name], list):
                for idx in range(1, len(dataset_row[input_arg_name]) + 1):
                    fields.update(
                        {
                            f"{input_arg_name}-{idx}": dataset_row[input_arg_name][
                                idx - 1
                            ].strip()
                        }
                    )
            else:
                fields.update({input_arg_name: dataset_row[input_arg_name]})
        suggestions = []
        for output_arg_name in self.output_args_names:
            if output_arg_name == "rating" and group_ratings_as_ranking:

                def ratings_as_ranking_value(ratings: List[int]):
                    indexed_ratings = list(enumerate(ratings, start=1))
                    sorted_ratings = sorted(
                        indexed_ratings, key=lambda x: x[1], reverse=True
                    )

                    ranked_fields = []
                    current_rank = 1
                    for i, (index, rating) in enumerate(sorted_ratings):
                        if i > 0 and rating < sorted_ratings[i - 1][1]:
                            current_rank = i + 1
                        ranked_fields.append(
                            {"rank": current_rank, "value": f"generations-{index}"}
                        )

                    return ranked_fields

                suggestions.append(
                    {
                        "question_name": "generations-ranking",
                        "value": ratings_as_ranking_value(dataset_row[output_arg_name]),
                    }
                )
                continue
            for idx, value in enumerate(dataset_row[output_arg_name], start=1):
                suggestions.append(
                    {
                        "question_name": f"generations-{idx}-{output_arg_name}",
                        "value": value.strip() if isinstance(value, str) else value,
                    }
                )
        return rg.FeedbackRecord(fields=fields, suggestions=suggestions)

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
