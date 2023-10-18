from typing import TYPE_CHECKING, List, Dict, Optional

from rlxf.rating_model import RatingModel
from rlxf.llm import HuggingFaceLLM as LLM

from datasets import Dataset

import argilla as rg

class PreferenceDataset:
    def __init__(self, dataset: Dataset, rating_model: Optional[RatingModel] = None, llm: Optional[LLM] = None, column_name: str = "text", num_responses: int = 2) -> None:
        self.dataset = dataset
        if llm is None and not self._dataset_has_responses():
            raise ValueError("If you don't pass an LLM, the dataset must contain a column named 'responses' containing the responses to be rated.")
        self.llm = llm
        self.rating_model = rating_model or RatingModel()
        self.column_name = column_name
        self.num_responses = num_responses 
        self.validate_dataset()

    def _dataset_has_responses(self):
        return "responses" in self.dataset.column_names

    def validate_dataset(self) -> None:
        if len(self.dataset) == 0:
            raise ValueError("The dataset is empty. Please provide a non-empty dataset.")
        if self.column_name not in self.dataset.column_names:
            raise ValueError(f"The required column '{self.column_name}' is not found in the dataset. "
                             f"Available columns: {self.dataset.column_names}")
        
    def generate(self, batch_size=1) -> Dict[str, List[str]]:
        def generate_responses(records) -> Dict[str, List[str]]:
            responses = self.llm.batch_generate(records[self.column_name])
            return {"responses": responses}
        if self.llm:
            # TODO: Improve batch processing
            generated_data = self.dataset.map(generate_responses, batched=True, batch_size=batch_size)
        else:
            generated_data = self.dataset

        # Get the ratings for the generated data
        rated_data = generated_data.map(
            self._generate_ratings,
        )

        ranked_data = rated_data.map(
            self._generate_rankings
        )

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
    
    def to_argilla(self) -> rg.FeedbackDataset:
        if not self._dataset_has_responses():
            raise ValueError("To convert to Argilla, the dataset must contain a column named 'responses'")
        # Configure input and response fields
        fields = [rg.TextField(name="input")]
        response_fields = [
            rg.TextField(name=f"response_{i}", use_markdown=True)
            for i in range(self.num_responses)
        ]
        fields.extend(response_fields)

        # Configure questions
        questions = []
        for i in range(self.num_responses):
            questions.extend([
                rg.RatingQuestion(
                    name=f"rating_{i}",
                    title=f"Rate the response_{i}?",
                    values=[1,2,3,4,5],
                ),
                rg.TextQuestion(
                    name=f"rationale_{i}",
                    title=f"Rationale behind response_{i}'s rating?",
                )
            ])
        # TODO: Configure metadata
        # add llm config, rating model config, text descriptives, rating average, etc.

        rg_dataset = rg.FeedbackDataset(
            fields=fields,
            questions=questions,
            #guidelines="", TODO: Define general guidelines template we can add here
        )

        # add records
        records = [
            self._build_argilla_record(example)
            for example in self.dataset
        ]
        rg_dataset.add_records(records)
        return rg_dataset

    def _build_argilla_record(self, example):
        # add field value for input
        fields = {"input": example[self.column_name]}

        # add field values for responses
        for i,r in enumerate(example["responses"]):
            fields[f"response_{i}"] = r

        # add suggestions
        suggestions = []
        # TODO: The name of the column for rating should be a constant defined for the whole project (it's created by the preference model)
        for i,feedback in enumerate(example["rating"]):
            suggestions.extend([
                { "question_name": f"rating_{i}", "value": int(feedback["rating"]), "agent": self.rating_model.config.model},
                { "question_name": f"rationale_{i}", "value": feedback["rationale"], "agent": self.rating_model.config.model},
            ])
        
        record = rg.FeedbackRecord(
            fields=fields,
            suggestions=suggestions
        )
        return record

    def estimate_cost(self) -> Dict[str, int]:
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
        return self.rating_model.rate_responses(record["responses"], record[self.column_name]) 
    
    def _generate_rankings(self, record):
        # Combine the two lists into a list of tuples and sort by rating in descending order
        combined = sorted([(d['rating'], i) for i, d in enumerate(record["rating"])], key=lambda x: x[0], reverse=True)

        # Generate the ranking string
        ranking = []
        prev_rating = None
        for i, (rating, index) in enumerate(combined):
            if prev_rating == rating:
                ranking.append(f"={index+1}")
            else:
                ranking.append(f">{index+1}")
            prev_rating = rating

        # Remove the first '>' symbol from the ranking string
        ranking[0] = ranking[0][1:]
        return {"ranking": ''.join(ranking)}