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

from rlxf.preference_dataset import PreferenceDataset
from rlxf.llm import LLM, LLMInferenceEndpoint

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient

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

# Configure local LLM using Transformers
llm = LLM(model=model, tokenizer=tokenizer, num_responses=2)

# or using HF Inference endpoints
# client = InferenceClient( "<HF_IE_URL>",  token="<HF_TOKEN>") 
# llm = LLMInferenceEndpoint(client=client, num_responses=4)

preference_dataset = PreferenceDataset(
    dataset, 
    llm=llm
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

Before first release:

- [ ] Make inference to generate responses more efficient (make using Mistral possible)
- [ ] Make GPT-4 rating more efficient (can we parallelize, batch this? add backoff, etc.) See related https://github.com/andrewgcodes/lightspeedGPT
- [x] Separate target model inference (the model used to generate responses)
- [x] Enable using HF inference endpoints instead of local model (nice to have)
- [x] Add to_argilla method 
- [x] Allow passing a dataset with generated responses and skip the generate responses step
- [ ] Cleanup, refactor code
- [ ] add tests
- [ ] show full example from generate to Argilla to DPO 
- [ ] final readme 

Later:
- [ ] Can we start rating without waiting for all responses to be generated? (nice to have)
- [ ] Add confidence rating in the prompt: how confident is the preference model about the ratings
- [ ] Compute Ranking from ratings
- [ ] Add metadata, text descriptives and measurements to metadata when doing `to_argilla()` to enable quick human curation.
- [ ] Add Ranking Model (do ranking instead of rating) (nice to have)
- [ ] Compute measurements about the quality/similarity of responses to filter those data points more useful for Rating and preference tuning.


