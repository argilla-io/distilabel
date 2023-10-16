# rlxf

Open-source framework for building preference datasets and models for LLM alignment.

## Usage

```python
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Instantiate the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

# Create a dataset
dataset = Dataset.from_dict({"text": ["My name is Daniel Vila Suero", "Please send me an email"]})

# Configure the RatingModel
rating_model_config = RatingModelConfig(openai_api_key="sk---")
rating_model = RatingModel(rating_model_config)

# Configure the PreferenceDataset
config = PreferenceDatasetConfig(num_responses=2, temperature=0.8)
preference_dataset = PreferenceDataset(dataset, model, tokenizer, rating_model, config)

# Execute methods for the PreferenceDataset
dry_run_output = preference_dataset.dry_run()
generated_data = preference_dataset.generate()
summary_info = preference_dataset.summary()

# Print or utilize the outputs as needed
print(dry_run_output)
print(generated_data)
print(summary_info)
```