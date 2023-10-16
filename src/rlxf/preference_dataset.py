from typing import List, Dict

from rlxf.rating_model import RatingModel
from rlxf.llm import LLM

from datasets import Dataset

class PreferenceDataset:
    def __init__(self, dataset:Dataset, rating_model: RatingModel=None, llm:LLM=None, column_name="text", num_responses=2):
        if llm is None and "responses" not in dataset.column_names:
            raise ValueError("If you don't pass an LLM, the dataset must contain a column named 'responses' containing the responses to be rated.")
        self.dataset = dataset
        self.llm = llm
        self.rating_model = rating_model or RatingModel()
        self.column_name = column_name
        self.num_responses = num_responses 
        self.validate_dataset()

    def validate_dataset(self) -> None:
        if len(self.dataset) == 0:
            raise ValueError("The dataset is empty. Please provide a non-empty dataset.")
        if self.column_name not in self.dataset.column_names:
            raise ValueError(f"The required column '{self.column_name}' is not found in the dataset. "
                             f"Available columns: {self.dataset.column_names}")
        
    def generate(self) -> Dict[str, List[str]]:
        def generate_responses(records) -> Dict[str, List[str]]:
            responses = self.llm.generate_responses(records[self.column_name])
            return {"responses": responses}
        if self.llm:
            generated_data = self.dataset.map(generate_responses)
        else:
            generated_data = self.dataset

        # Get the ratings for the generated data
        rated_data = generated_data.map(
            self._generate_ratings,
        )
        return rated_data

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

    def _generate_ratings(self, record: Dict[str, List[str]]) -> Dict[str, List[float]]:
        return self.rating_model.rate_responses(record["responses"], record[self.column_name])  