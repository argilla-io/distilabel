<h1 align="center">
  <a href=""><img src="docs/rlxf.png" alt="rlxf image" width="30%"></a>
  <br>
  ✨ RLxF for building preference datasets ✨
  <br>
</h1>

## What's RLxF
An open-source framework for building preference datasets for LLM alignment.

## Usage

```python
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup openai api key to use GPT4 Rating Model
os.environ['OPENAI_API_KEY'] = 'sk-***'

# Instantiate the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
#tokenizer = AutoTokenizer.from_pretrained("PY007/TinyLlama-1.1B-Chat-v0.3")
#model = AutoModelForCausalLM.from_pretrained("PY007/TinyLlama-1.1B-Chat-v0.3")

# Create a dataset
dataset = Dataset.from_dict(
    {"text": [
        "Write an email for B2B marketing: ##EMAIL: ", 
        "What is the name of the capital of France? ",
    ]}
)

# Configure the RatingModel
rating_model = RatingModel()

# Configure the LLM
llm = LLM(model=model, tokenizer=tokenizer, num_responses=2)

preference_dataset = PreferenceDataset(
    dataset, 
    llm=llm, 
    rating_model=rating_model
)

# Methods
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

- [x] Separate target model inference (the model used to generate responses)
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
