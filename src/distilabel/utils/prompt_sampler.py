import random
from pathlib import Path as pth 
import json

from distilabel.pydantics import CategoricalDist, PromptSamplerConfig

class PromptSampler:
    '''
    This is somewhat complicated, you have a template system prompt with kwargs to sample, but I want the system prompt 
    to handle asking the model for multiple questions at once and sampling the kwargs for each question. This class allows
    one of the sampled kwargs to specify a count for other kwargs that need to be sampled multiple times, then samples those 
    kwargs that many times, putting them in a list and potentially grouping them side by side so that the requirements for each 
    output are in one list, next to each other in the system prompt.

    Also conveniently you can sample text files and insert their contents (for long instructions).

    If you put an empty string in a side by side kwarg, it will be removed from the list.
    '''
    def __init__(self, cfg: PromptSamplerConfig, template: str):
        self.cfg = cfg
        self.template = template

    def sample_kwarg(self, dist: CategoricalDist):
        kwargs = []
        for i in range(dist.samples_per_prompt):
            values, probabilities = zip(*dist.choices)
            sampled_value = random.choices(values, probabilities)[0]
            if pth(sampled_value).exists() and pth(sampled_value).suffix == ".txt":
                sampled_value = pth(sampled_value).read_text()
            kwargs.append(sampled_value)
        if dist.samples_per_prompt == 1:
            return kwargs[0]
        return kwargs

    def sample(self):
        '''Sample a prompt according to the prompt sampler config'''
        dists = self.cfg.distributions
        if self.cfg.samples_per_prompt_kwarg:
            assert dists[self.cfg.samples_per_prompt_kwarg].samples_per_prompt is not None, (
                "samples_per_prompt_kwarg must have a defined samples_per_prompt"
            )
        # break into two groups, post-requisites and pre-requisites
        # post-requisites require a sampled value for samples_per_prompt (like, generate 1-5 questions)
        post_req_kwargs = {k for k in dists.keys() if dists[k].samples_per_prompt is None}
        prereq_kwargs = {k for k in dists.keys() if k not in post_req_kwargs}

        # sample pre-requisites
        kwargs = {}
        for kwarg in prereq_kwargs:
            kwargs[kwarg] = self.sample_kwarg(dists[kwarg])

        # sample post-requisites, pulling the samples_per_prompt from the pre-requisites[samples_per_prompt_kwarg]
        post_req_samples_per_prompt = int(kwargs.get(self.cfg.samples_per_prompt_kwarg, 1))
        if len(post_req_kwargs) > 0:
            for kwarg in post_req_kwargs:
                dists[kwarg].samples_per_prompt = post_req_samples_per_prompt
                kwargs[kwarg] = self.sample_kwarg(dists[kwarg])

        # this is just formatting, if you want to say generate a list of questions, you may want to also specify details of the question for each,
        # then, you wouldn't want to have two sections of the system prompt where the first says what type for each and the second says what 
        # word to start with, you'd want a list where each entry says what type and what word, that's what this does
        side_by_side_kwargs = [k for k in dists.keys() if dists[k].side_by_side]
        broadcasted_kwargs = []
        for kwarg in side_by_side_kwargs:
            sampled_value = kwargs.pop(kwarg)
            if isinstance(sampled_value, list):
                broadcasted_kwargs.append(sampled_value)
            else:
                broadcasted_kwargs.append([sampled_value] * post_req_samples_per_prompt)
        kwargs['side_by_side'] = list(zip(*broadcasted_kwargs))
        
        # remove empty strings from lists
        for k, v in kwargs.items():
            if isinstance(v, list):
                kwargs[k] = [i for i in v if i != ""]
        kwargs = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in kwargs.items()}
        # now has each normal one ready to insert, side by side ones broadcasted and all list likes jsonified 
        return kwargs

    def generate_prompt(self) -> str:
        '''Format the template with the sampled kwargs'''
        return self.template.format(**self.sample())


