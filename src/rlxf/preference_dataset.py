class PreferenceDataset:
    def __init__(self, dataset, model, tokenizer, rating_model, config=None):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or PreferenceDatasetConfig()
        self.rating_model = rating_model  
        self.validate_dataset()

    def validate_dataset(self):
        if len(self.dataset) == 0:
            raise ValueError("The dataset is empty. Please provide a non-empty dataset.")
        if self.config.column_name not in self.dataset.column_names:
            raise ValueError(f"The required column '{self.config.column_name}' is not found in the dataset. "
                             f"Available columns: {self.dataset.column_names}")

    def _validate_input(self, text):
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text)}")

    def _generate_responses(self, text):
        self._validate_input(text)
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=self.config.max_length,
            num_return_sequences=self.config.num_responses,
            num_beams=max(2, self.config.num_responses),
            temperature=self.config.temperature,
            do_sample=True,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
        )
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return responses

    def dry_run(self):
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


    def generate(self):
        def generate_responses(records):
            responses = self._generate_responses(records[self.config.column_name])
            return {"responses": responses}

        generated_data = self.dataset.map(generate_responses)

        # Get the ratings for the generated data
        rated_data = generated_data.map(
            self._generate_ratings,
        )

        return rated_data

    def estimate_cost(self):
        return {"num_data_points": n, "num_tokens": 111, "estimated_cost": 111}

    def summary(self):
        summary_info = {
            "Dataset Size": len(self.dataset),
            "Number of Responses": self.config.num_responses,
            "Temperature": self.config.temperature,
            # ... Add other relevant information
        }
        return summary_info

    def _generate_ratings(self, record):
        return self.rating_model.rate_responses(record["responses"])  

class PreferenceDatasetConfig:
    def __init__(self, num_responses=2, temperature=0.7, column_name="text", max_length=50,
                 top_k=50, top_p=0.95, repetition_penalty=1.2, no_repeat_ngram_size=2, **kwargs):
        self.num_responses = num_responses
        self.temperature = temperature
        self.column_name = column_name
        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.extra_args = kwargs  # For any extra arguments

# Usage:
# rating_model_config = RatingModelConfig()
# rating_model = RatingModel(rating_model_config)
# config = PreferenceDatasetConfig(num_responses=3, temperature=0.8)
# preference_dataset = PreferenceDataset(dataset, model, tokenizer, rating_model, config)
# dry_run_output = preference_dataset.dry_run()
# generated_data = preference_dataset.generate()
# summary_info = preference_dataset.summary()
