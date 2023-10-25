from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from datasets import Dataset

from rlxf.rating_model import RatingModel

if TYPE_CHECKING:
    from rlxf.llm.base import LLM

try:
    import argilla as rg

    _argilla_installed = True
except ImportError:
    _argilla_installed = False


if TYPE_CHECKING and _argilla_installed:
    from argilla import FeedbackDataset


class PreferenceDataset:
    def __init__(
        self,
        dataset: Dataset,
        rating_model: Optional[RatingModel] = None,
        llm: Optional[LLM] = None,
        column_name: str = "text",
        num_responses: int = 2,
    ) -> None:
        self.dataset = dataset

        if llm is None and not self._dataset_has_column("responses"):
            raise ValueError(
                "If you don't pass an LLM, the dataset must contain a column named 'responses' containing the responses to be rated."
            )
        self.llm = llm

        self.rating_model = rating_model or RatingModel(num_responses=num_responses)
        if num_responses != self.rating_model.num_responses:
            raise ValueError(
                f"The number of responses must match the number of responses expected by the rating model: {self.rating_model.num_responses}"
            )

        self.column_name = column_name
        self.num_responses = num_responses
        self.validate_dataset()

    def _dataset_has_column(self, column_name):
        return column_name in self.dataset.column_names

    def validate_dataset(self) -> None:
        if len(self.dataset) == 0:
            raise ValueError(
                "The dataset is empty. Please provide a non-empty dataset."
            )
        if self.column_name not in self.dataset.column_names:
            raise ValueError(
                f"The required column '{self.column_name}' is not found in the dataset. "
                f"Available columns: {self.dataset.column_names}"
            )

    def generate(self, batch_size=1) -> Dict[str, List[str]]:
        def generate_responses(records) -> Dict[str, List[str]]:
            responses = self.llm.batch_generate(records[self.column_name])
            return {"responses": responses}

        if self.llm:
            # TODO: Improve batch processing
            generated_data = self.dataset.map(
                generate_responses, batched=True, batch_size=batch_size
            )
        else:
            generated_data = self.dataset

        # Get the ratings for the generated data
        rated_data = generated_data.map(
            self._generate_ratings,
        )

        ranked_data = rated_data.map(self._generate_rankings)

        # Update the dataset with the generated data
        self.dataset = ranked_data

        return ranked_data

    def dry_run(self) -> Dict[str, List[str]]:
        # Create a subset of the dataset with 1 or 2 records
        test_dataset = self.dataset.select(range(1))  # Adjust the range as needed

        # Replace the original dataset with the test dataset
        original_dataset = self.dataset
        self.dataset = test_dataset

        try:
            # Call the generate method to test the full workflow
            generated_data = self.generate()
        finally:
            # Restore the original dataset
            self.dataset = original_dataset

        return generated_data

    def to_argilla(self, use_ranking: bool = False) -> "FeedbackDataset":
        if not _argilla_installed:
            raise ImportError(
                "In order to use the `to_argilla` method, you need to install Argilla via"
                " `pip install argilla` or `pip install rlxf[argilla]`."
            )
        if not self._dataset_has_column("responses") or not self._dataset_has_column(
            "rating"
        ):
            raise ValueError(
                f"To convert to Argilla, the dataset must contain at least 'responses' and 'rating' columns. Current columns: {self.dataset.column_names}"
            )

        # Configure input and response fields
        fields = [rg.TextField(name="input")]
        response_fields = [
            rg.TextField(name=f"response_{i}", use_markdown=True)
            for i in range(1, self.num_responses + 1)
        ]
        fields.extend(response_fields)

        # Configure questions
        questions = []
        if use_ranking:
            # for ranking we define just one question
            preference_question = rg.RankingQuestion(
                name="ranking",
                title="Rank the responses",
                values=[f"response_{i}" for i in range(1, self.num_responses + 1)],
            )
            questions.append(preference_question)
            for i in range(1, self.num_responses + 1):
                questions.extend([self._build_rationale_question(i)])
        else:
            for i in range(1, self.num_responses + 1):
                # for rating we need one rating per response
                questions.extend(
                    [
                        rg.RatingQuestion(
                            name=f"rating_{i}",
                            title=f"Rate the response_{i}?",
                            values=[1, 2, 3, 4, 5],
                        ),
                        self._build_rationale_question(i),
                    ]
                )

        # TODO: Configure metadata
        # add llm config, rating model config, text descriptives, rating average, etc.

        rg_dataset = rg.FeedbackDataset(
            fields=fields,
            questions=questions,
            # guidelines="", TODO: Define general guidelines template we can add here
        )

        # add records
        records = [
            self._build_argilla_record(example, use_ranking=use_ranking)
            for example in self.dataset
        ]
        rg_dataset.add_records(records)
        return rg_dataset

    def _build_rationale_question(self, i):
        return rg.TextQuestion(
            name=f"rationale_{i}",
            title=f"Rationale behind response_{i}'s ranking?",
        )

    def _build_argilla_record(self, example, use_ranking):
        # add field value for input
        fields = {"input": example[self.column_name]}

        # add field values for responses
        for i, r in enumerate(example["responses"]):
            fields[f"response_{i+1}"] = r

        # Add suggestions for ranking or rating
        # TODO: The name of the columns rating/ranking should be a constant defined for the whole project (it's created by the preference model)

        suggestions = []
        if use_ranking:
            suggestions.append(self._build_ranking_suggestion(example["ranking"]))
            for i, feedback in enumerate(example["rating"]):
                suggestions.extend(
                    [
                        self._build_rationale_suggestions(i + 1, feedback),
                    ]
                )
        else:
            for i, feedback in enumerate(example["rating"]):
                suggestions.extend(
                    [
                        {
                            "question_name": f"rating_{i+1}",
                            "value": int(feedback["rating"]),
                            "agent": self.rating_model.model,
                        },
                        self._build_rationale_suggestions(i + 1, feedback),
                    ]
                )

        # then add suggestions
        record = rg.FeedbackRecord(fields=fields, suggestions=suggestions)
        return record

    def _build_rationale_suggestions(self, i, feedback):
        return {
            "question_name": f"rationale_{i}",
            "value": feedback["rationale"],
            "agent": self.rating_model.model,
        }

    def _build_ranking_suggestion(self, ranking_str):
        parts = ranking_str.split(">")
        rank_values = []
        rank = 1

        for part in parts:
            tied_indices = [int(i) for i in part.split("=")]
            for i in tied_indices:
                rank_values.append({"rank": rank, "value": f"response_{i}"})

            rank += 1

        return {"question_name": "ranking", "value": rank_values}

    def estimate_cost(self) -> Dict[str, int]:
        # TODO: Estimate OpenAI API costs
        n = len(self.dataset)
        return {"num_data_points": n, "num_tokens": 111, "estimated_cost": 111}

    def summary(self) -> Dict[str, int]:
        summary_info = {
            "Dataset Size": len(self.dataset),
            "Number of Responses": self.num_responses,
            "Temperature": self.llm.temperature,
            # ... Add other relevant information
        }
        return summary_info

    def _generate_ratings(self, record):
        return self.rating_model.rate_responses(
            record["responses"], record[self.column_name]
        )

    def _generate_rankings(self, record):
        # Convert ratings to integers and combine the two lists into a list of tuples.
        # Then sort by rating in descending order.
        combined = sorted(
            [(int(d["rating"]), i) for i, d in enumerate(record["rating"])],
            key=lambda x: x[0],
            reverse=True,
        )

        # Generate the ranking string
        ranking = []
        prev_rating = None
        for _, (rating, index) in enumerate(combined):
            if prev_rating == rating:
                ranking.append(f"={index+1}")
            else:
                ranking.append(f">{index+1}")
            prev_rating = rating

        # Remove the first '>' symbol from the ranking string
        ranking[0] = ranking[0][1:]
        return {"ranking": "".join(ranking)}
