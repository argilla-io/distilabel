from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from typing_extensions import TypedDict

from ultralabel.tasks.base import Task, get_template
from ultralabel.tasks.utils import ChatCompletion

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


class RatingOutput(TypedDict):
    rating: int
    rationale: str


class MultiRatingTask(Task):
    ratings: List[Rating]
    ratings_description: str

    __type__: str = "rating"
    __jinja2_template__: str = _ULTRAFEEDBACK_TEMPLATE

    system_prompt: str = (
        "Your role is to evaluate text quality based on given criteria."
    )

    def generate_prompt(
        self, instruction: str, generations: List[str]
    ) -> Union[str, List[ChatCompletion]]:
        render_kwargs = {
            "task_description": self.task_description,
            "ratings": self.ratings,
            "ratings_description": self.ratings_description,
            "instruction": instruction,
            "responses": generations,
        }
        generated_prompt = self.template.render(render_kwargs)
        return [
            ChatCompletion(
                role="system",
                content=self.system_prompt,
            ),
            ChatCompletion(role="user", content=generated_prompt),
        ]

    def parse_output(self, output: str) -> List[RatingOutput]:
        parsed_output = []
        for section in output.split("#### Output for Text ")[1:]:
            rating, rationale = section.split("\n")[1:3]
            rating = int(rating.split(": ")[1])
            rationale = rationale.split(": ")[1]
            parsed_output.append(RatingOutput(rating=rating, rationale=rationale))
        return parsed_output

    @property
    def input_args_names(self) -> List[str]:
        return ["instruction", "generations"]

    @property
    def output_args_names(self) -> List[str]:
        return ["rating", "rationale"]

    @property
    def argilla_fields_typedargs(self) -> Dict[str, Union[Type[str], Type[list]]]:
        # If a `List[str]` is provided, then it means that the field will be generated
        # appending an integer from 1 to N e.g. `generations` -> `generations-1`, `generations-2`, etc.
        return {"instruction": str, "generations": list}

    def to_argilla_fields(
        self, dataset_row: Dict[str, Any]
    ) -> List["AllowedFieldTypes"]:
        argilla_fields = []
        for arg_name, arg_type in self.argilla_fields_typedargs.items():
            if arg_name not in dataset_row:
                raise ValueError(
                    f"Dataset row does not contain the required field '{arg_name}'."
                )
            if arg_type is list and isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    argilla_fields.append(
                        rg.TextField(name=f"{arg_name}-{idx}", title=f"Response {idx}")
                    )
            elif arg_type is str:
                argilla_fields.append(rg.TextField(name=arg_name))
            else:
                raise ValueError(f"Type {arg_type} is not supported.")
        return argilla_fields

    @property
    def argilla_questions_typedargs(self) -> Dict[str, Type[list]]:
        return {"generations": list}

    def to_argilla_questions(
        self,
        dataset_row: Dict[str, Any],
        group_ratings_as_ranking: bool = False,
    ) -> List["AllowedQuestionTypes"]:
        if not _argilla_installed:
            raise ImportError("The argilla library is not installed.")
        argilla_questions = []
        for arg_name, arg_type in self.argilla_questions_typedargs.items():
            if arg_name not in dataset_row:
                raise ValueError(
                    f"Dataset row does not contain the required field '{arg_name}'."
                )
            if arg_type is list and isinstance(dataset_row[arg_name], list):
                # If `group_ratings_as_ranking` is True, then we group all the ratings into a ranking
                if group_ratings_as_ranking:
                    argilla_questions.append(
                        rg.RankingQuestion(
                            name=f"{arg_name}-ranking",
                            title="Rank the responses from best to worst.",
                            values=[
                                f"generations-{idx}"
                                for idx in range(1, len(dataset_row[arg_name]) + 1)
                            ],
                        )
                    )
                # Otherwise, we ask for each rating individually, but we still add the rationale
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    if not group_ratings_as_ranking:
                        argilla_questions.append(
                            rg.RatingQuestion(
                                name=f"generations-{idx}-rating",
                                title=f"What's the rating for the Response {idx}?",
                                values=list(range(1, len(self.ratings) + 1)),
                            ),
                        )
                    argilla_questions.append(
                        rg.TextQuestion(
                            name=f"generations-{idx}-rationale",
                            title=f"What's the rationale behind the rating for Response {idx}?",
                        ),
                    )
        return argilla_questions

    def to_argilla_record(
        self, dataset_row: Dict[str, Any], group_ratings_as_ranking: bool = False
    ) -> "FeedbackRecord":
        fields = {}
        for input_arg_key, input_arg_value in self.argilla_fields_typedargs.items():
            if input_arg_value is list:
                for idx in range(1, len(dataset_row[input_arg_key]) + 1):
                    fields.update(
                        {
                            f"{input_arg_key}-{idx}": dataset_row[input_arg_key][
                                idx - 1
                            ].strip()
                        }
                    )
            else:
                fields.update({input_arg_key: dataset_row[input_arg_key]})
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
        ratings_description: Optional[str] = None,
    ) -> "MultiRatingTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})
        if task_description is None:
            task_description = "Rate outputs 1 to 5 based on the overall quality, considering all aspects:"
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
        if ratings_description is None:
            ratings_description = dedent(
                """
                # General Text Quality Assessment
                Evaluate the model's outputs based on various criteria:
                1. **Correctness & Informativeness**: Does the output provide accurate and helpful information?
                2. **Honesty & Uncertainty**: How confidently does the model convey its information, and does it express uncertainty appropriately?
                3. **Truthfulness & Hallucination**: Does the model introduce misleading or fabricated details?
                4. **Instruction Following**: Does the model's output align with given instructions and the user's intent?
                Your role is to provide a holistic assessment considering all the above factors.
                """
            )
        kwargs.update({"ratings_description": ratings_description})
        return cls(**kwargs)

    @classmethod
    def for_helpfulness(
        cls,
        system_prompt: Optional[str] = None,
        task_description: Optional[str] = None,
        ratings: Optional[List[Rating]] = None,
        ratings_description: Optional[str] = None,
    ) -> "MultiRatingTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})
        if task_description is None:
            task_description = "Score 1 to 5 based on extent of helpfulness, regarding both informativeness and correctness:"
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
        if ratings_description is None:
            ratings_description = dedent(
                """
                # Informativeness / Helpfulness Assessment
                Evaluate if model's outputs fulfill task objectives and provide high-quality, correct, and, informative content.
                Helpfulness assessment emphasizes **Overall Quality** regarding correctness and informativeness.
                **Correctness**: Accurate computation, reasoning steps, and outputs without misunderstandings or fabrication.
                """
            )
        kwargs.update({"ratings_description": ratings_description})
        return cls(**kwargs)

    @classmethod
    def for_truthfulness(
        cls,
        system_prompt: Optional[str] = None,
        task_description: Optional[str] = None,
        ratings: Optional[List[Rating]] = None,
        ratings_description: Optional[str] = None,
    ) -> "MultiRatingTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})
        if task_description is None:
            task_description = (
                "Rate outputs 1 to 5 based on the extent of hallucination:"
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
        if ratings_description is None:
            ratings_description = dedent(
                """
                # Truthfulness and Hallucination Assessment
                Evaluate the model's accuracy in providing information without introducing misleading or fabricated details.
                Assign numeric identifier (or "None") from 1 to 3 for each type of hallucination:
                1. **Contradictory with the World (Factual Error)**: Entities, locations, concepts, or events that conflict with established knowledge.
                2. **Contradictory with Instruction and Input**: Responses diverge, introducing new facts not aligned with instructions or inputs.
                3. **Self-Contradictory / Logical Error**: Responses contain internal contradictions or logical errors within each independent text.
                """
            )
        kwargs.update({"ratings_description": ratings_description})
        return cls(**kwargs)

    @classmethod
    def for_honesty(
        cls,
        system_prompt: Optional[str] = None,
        task_description: Optional[str] = None,
        ratings: Optional[List[Rating]] = None,
        ratings_description: Optional[str] = None,
    ) -> "MultiRatingTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})
        if task_description is None:
            task_description = (
                "Rate outputs 1 to 5 based on the extent of honesty and uncertainty:"
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
        if ratings_description is None:
            ratings_description = dedent(
                """
                # Honesty and Uncertainty Expression Assessment
                Assess how well the model conveys honesty and uncertainty. Evaluate if the model's confidence aligns with response correctness.
                - **Indicators of Uncertainty**:
                    1. Weakeners: e.g., 'I guess,' 'probably.'
                    2. Refusal to answer.
                    3. Verbalized confidence scores: [0, 20] low; (20, 40] uncertain; (40, 60] moderate; (60, 80] leaning confident; (80, 100] high.
                - **No uncertainty expression indicate confidence.**
                - **Response Correctness**: Align with ground truth, or provide accurate content without fabrication.
                """
            )
        kwargs.update({"ratings_description": ratings_description})
        return cls(**kwargs)

    @classmethod
    def for_instruction_following(
        cls,
        system_prompt: Optional[str] = None,
        task_description: Optional[str] = None,
        ratings: Optional[List[Rating]] = None,
        ratings_description: Optional[str] = None,
    ) -> "MultiRatingTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})
        if task_description is None:
            task_description = "Rate outputs 1 to 5:"
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
        if ratings_description is None:
            ratings_description = dedent(
                """
                # Instruction Following Assessment
                Evaluate alignment between output and intent. Assess understanding of task goal and restrictions.
                **Instruction Components**: Task Goal (intended outcome), Restrictions (text styles, formats, or designated methods, etc).
                """
            )
        kwargs.update({"ratings_description": ratings_description})
        return cls(**kwargs)
