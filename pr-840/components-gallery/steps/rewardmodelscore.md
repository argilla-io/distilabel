---
hide:
  - navigation
---
# RewardModelScore

Assign a score to a response using a Reward Model.



`RewardModelScore` is a `Step` that using a Reward Model (RM) loaded using `transformers`,
    assigns an score to a response generated for an instruction, or a score to a multi-turn
    conversation.





### Attributes

- **model**: the model Hugging Face Hub repo id or a path to a directory containing the  model weights and configuration files.

- **revision**: if `model` refers to a Hugging Face Hub repository, then the revision  (e.g. a branch name or a commit id) to use. Defaults to `"main"`.

- **torch_dtype**: the torch dtype to use for the model e.g. "float16", "float32", etc.  Defaults to `"auto"`.

- **trust_remote_code**: whether to allow fetching and executing remote code fetched  from the repository in the Hub. Defaults to `False`.

- **device_map**: a dictionary mapping each layer of the model to a device, or a mode like `"sequential"` or `"auto"`. Defaults to `None`.

- **token**: the Hugging Face Hub token that will be used to authenticate to the Hugging  Face Hub. If not provided, the `HF_TOKEN` environment or `huggingface_hub` package  local configuration will be used. Defaults to `None`.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
	end

	subgraph RewardModelScore
	end


```









