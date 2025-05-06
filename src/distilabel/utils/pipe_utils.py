import random
from typing import Callable

from distilabel.pipeline import routing_batch_function
from distilabel.steps import Step

from distilabel.models.llms import OpenAILM
from distilabel.pydantics import Stage

def steps_to_load_groups(steps: list[Step], n_gpus: int) -> list[list[str]]:
    '''
    Given a list of steps in an steps, break the steps into load_groups so that there are enough gpus for each load_group
    
    Return a list of lists of step names, where each inner list is a load_group in the pipeline
    '''
    load_groups = []
    load_group = []
    load_group_gpus = 0
    for step in steps:
        if not hasattr(step, 'resources') or step.resources.gpus is None:
            load_group.append(step)
            continue
        requested_gpus = step.resources.gpus * step.resources.replicas
        assert requested_gpus <= n_gpus
        if (load_group_gpus + requested_gpus) > n_gpus:
            load_groups.append(load_group)
            load_group = [step.name]
            load_group_gpus = 0
        else:
            load_group.append(step.name)
            load_group_gpus += requested_gpus
    load_groups.append(load_group)
    return load_groups

def data_router(step_distribution: list[float]) -> Callable:
    '''
    Given a list of downstream steps and a distribution, per batch, sample the downstream step to route to

    Use this as a step in a pipeline to e.g. split the data from a step to different language models
    '''
    @routing_batch_function()
    def router(steps: list[str]) -> list[str]:
        return random.choices(steps, weights=step_distribution, k=1)
    return router

def make_lms(stage: Stage) -> list[OpenAILM]:
    '''initialize lms for a stage'''
    return [
        OpenAILM(
            stage=stage, 
            lm_config=lm_config, 
            model=lm_config.path, 
            generation_kwargs={
                'temperature': lm_config.temperature, 
                'max_new_tokens': lm_config.max_new_tokens,
            },
        ) 
        for lm_config in stage.lm_configs
    ]

