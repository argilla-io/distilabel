# rlaif

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
preference_dataset = PreferenceDataset(dataset, model, tokenizer, rating_model)

# Execute methods for the PreferenceDataset
dry_run_output = preference_dataset.dry_run()
generated_data = preference_dataset.generate()
summary_info = preference_dataset.summary()
rg_dataset = preference_dataset.to_argilla()
# Print or utilize the outputs as needed
print(dry_run_output)
print(generated_data)
print(summary_info)
```

## TODOS

- [ ] Separate target model inference (the model used to generate responses
- [ ] Make inference to generate responses more efficient (make using Mistral possible)
- [ ] Enable using inference endpoints instead of local model (nice to have)
- [ ] Make gpt rating more efficient (can we parallelize this)
- [ ] Can we start rating without waiting for all responses to be generated (nice to have)
- [ ] Add Ranking Model (do ranking instead of rating) (nice to have)
- [ ] Add to_argilla method 
- [ ] Allow passing a dataset with generated responses and skip the generate responses step
- [ ] Cleanup, refactor code
- [ ] add tests
- [ ] show full example from generate to Argilla to DPO 
- [ ] final readme 
