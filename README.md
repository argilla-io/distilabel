 <div align="center">
   <h1>⚗️ distilabel</h1>
   <p>
     <em>Distilling datasets with LLMs</em>
   </p>
 </div>

## What's distilabel
distilabel is a framework for building datasets and labelers powered by LLMs.


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


