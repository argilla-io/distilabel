<h1 align="center">
  <a href=""><img src="docs/ultralabel.png" alt="rlxf image" width="100%"></a>
  <br>
  Ultralabel: your AI labeller for LLMs
  <br>
</h1>

## What's Ultralabel

## Usage

## With a dataset containing responses

```python
from rlxf.preference_dataset import PreferenceDataset
from datasets import load_dataset

# Setup openai api key to use GPT4 Rating Model
os.environ['OPENAI_API_KEY'] = 'sk-***'

dataset = load_dataset("argilla/mistral_vs_llama2", split="train")

pd = PreferenceDataset(
    dataset=dataset,
    num_responses=2
)

pd.generate()
```

## Local model with `transformers`
```python
import os

from rlxf.preference_dataset import PreferenceDataset
from rlxf.llm import LLM

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup openai api key to use GPT4 Rating Model
os.environ['OPENAI_API_KEY'] = 'sk-***'

# Instantiate the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
#tokenizer = AutoTokenizer.from_pretrained("PY007/TinyLlama-1.1B-Chat-v0.3")
#model = AutoModelForCausalLM.from_pretrained("PY007/TinyLlama-1.1B-Chat-v0.3")

# Read or create source dataset
dataset = Dataset.from_dict(
    {"text": [
        "Write an email for B2B marketing: ##EMAIL: ", 
        "What is the name of the capital of France? ",
    ]}
)

# Configure local LLM using Transformers
llm = LLM(model=model, tokenizer=tokenizer, num_responses=4)

pd = PreferenceDataset(
    dataset, 
    llm=llm,
    num_responses=4
)

# make sure everything is working
pd.dry_run()

pd.generate()
```

## Using Inference endpoints

```python
import os

from rlxf.preference_dataset import PreferenceDataset
from rlxf.llm import LLM

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup openai api key to use GPT4 Rating Model
os.environ['OPENAI_API_KEY'] = 'sk-***'

# Setup hf ie client
client = InferenceClient(
    "<HF URL>", 
    token="<HF_API>"
)

llm = LLMInferenceEndpoint(client=client, num_responses=4)

# Read or create source dataset
dataset = Dataset.from_dict(
    {"text": [
        "Write an email for B2B marketing: ##EMAIL: ", 
        "What is the name of the capital of France? ",
    ]}
)

pd = PreferenceDataset(
    dataset, 
    llm=llm,
    num_responses=4
)

# make sure everything is working
pd.dry_run()

# generate preference dataset
pd.generate()
```

## Load dataset into Argilla

```python

# set use_ranking to false to use rating for each response instead of ranking
dataset = pd.to_argilla(use_ranking=True)
dataset.push_to_argilla(name="mistral_preference_dataset", workspace="admin")

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


