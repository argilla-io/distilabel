import dotenv
dotenv.load_dotenv()

from distilabel.pydantics import (
    Config, 
    Stage, 
    LMConfig, 
    PromptSamplerConfig,
    CategoricalDist,
)
from distilabel import utils

question_words = [
    'What', 
    'Who', 
    'Where', 
    'When', 
    'Why', 
    'How', 
    'Which', 
    'Do', 
    'Does', 
    'Is', 
    'Are', 
    'Has', 
    'Have', 
    'Will', 
    'Would', 
    'Can', 
    'Should',
]
start_question_with = [
    (f'begin your question with "{word}" (translated into the language of the page)', 1) 
    for word in question_words
]

question_prompt_sampler_config = PromptSamplerConfig(
    samples_per_prompt_kwarg='n_questions',
    distributions={
        'n_questions': CategoricalDist(
            choices=[('1', 1), ('2', 1), ('3', 1), ('4', 1), ('5', 1)]
        ),
        'question_spec': CategoricalDist(
            choices=[
                ("pick a section of the page and ask for a summary of that section", 1),
                ("ask for a summary of the entire page", 1),
                ("ask a question requiring comprehension of a specific section of the page", 1),
                ("ask a question requiring comprehension of the entire page and ask for a detailed response", 1),
                ("ask a question requiring multi-step reasoning about the page and ask for the model's thought process", 1),
                ("ask a question that requires an open ended answer and ask for a detailed response", 1),
                ("request a specific piece of information from the page", 1)
            ],
            samples_per_prompt=None,
            side_by_side=True,
        ),
        'start_question_with': CategoricalDist(
            choices=start_question_with + [("", 16)],
            samples_per_prompt=None,
            side_by_side=True,
        ),
        'side_by_side_prefix': CategoricalDist(
            choices=[
                ("Given the number of questions to generate, the following are guidelines for each question in particular:", 1.0)
            ],
        )
    }
)
answer_prompt_sampler_config = PromptSamplerConfig()

AVAILABLE_GPUS = [4, 5]
task_name = 'question_generation'
data_ratios = utils.normalize_distribution([1] * 1)
stages = [
    Stage(
        lm_configs=[
            LMConfig(
                path='Qwen/Qwen2-VL-7B-Instruct', 
                data_ratio=data_ratios[0], 
                task_name=task_name,
                temperature=0.4,
                max_new_tokens=2048,
                tp_size=1,
                replicas=2,
                out_model='SinglePageQuestions',
                prompt_sampler_config=question_prompt_sampler_config,
            ),
            # LMConfig(
            #     path='gpt-4.1-mini', 
            #     data_ratio=data_ratios[1], 
            #     task_name=task_name,
            #     replicas=1,
            #     out_model='SinglePageQuestions',
            #     prompt_sampler_config=question_prompt_sampler_config,
            # ),
            # LMConfig(
            #     path='gemini-2.0-flash-lite',
            #     data_ratio=data_ratios[2],
            #     task_name=task_name,
            #     replicas=1,
            #     out_model='SinglePageQuestions',
            #     prompt_sampler_config=question_prompt_sampler_config,
            # ),
            # LMConfig(
            #     path='claude-3-haiku-20240307', 
            #     data_ratio=data_ratios[3], 
            #     task_name=task_name,
            #     replicas=1,
            #     out_model='SinglePageQuestions',
            #     prompt_sampler_config=question_prompt_sampler_config,
            # ),
            # LMConfig(
            #     path='grok-3-mini-beta', 
            #     data_ratio=data_ratios[4], 
            #     task_name=task_name,
            #     replicas=1,
            #     out_model='SinglePageQuestions',
            #     prompt_sampler_config=question_prompt_sampler_config,
            # ),
        ],
        default_system_template_path='distilabel/prompts/single_page_questions.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(512, 512),
    ),
    Stage(
        lm_configs=[
            LMConfig(
                path='Qwen/Qwen2-VL-7B-Instruct', 
                data_ratio=data_ratios[0], 
                task_name=task_name,
                tp_size=1,
                replicas=2,
                out_model='SinglePageAnswers',
                prompt_sampler_config=answer_prompt_sampler_config,
            ),
            # LMConfig(
            #     path='gpt-4.1-mini', 
            #     data_ratio=data_ratios[1], 
            #     task_name=task_name,
            #     replicas=1,
            #     out_model='SinglePageAnswers',
            #     prompt_sampler_config=answer_prompt_sampler_config,
            # ),
            # LMConfig(
            #     path='gemini-2.0-flash-lite',
            #     data_ratio=data_ratios[2],
            #     task_name=task_name,
            #     replicas=1,
            #     out_model='SinglePageAnswers',
            #     prompt_sampler_config=answer_prompt_sampler_config,
            # ),
            # LMConfig(
            #     path='claude-3-haiku-20240307', 
            #     data_ratio=data_ratios[3], 
            #     task_name=task_name,
            #     replicas=1,
            #     out_model='SinglePageAnswers',
            #     prompt_sampler_config=answer_prompt_sampler_config,
            # ),
            # LMConfig(
            #     path='grok-3-mini-beta', 
            #     data_ratio=data_ratios[4], 
            #     task_name=task_name,
            #     replicas=1,
            #     out_model='SinglePageAnswers',
            #     prompt_sampler_config=answer_prompt_sampler_config,
            # ),
        ],
        default_system_template_path='distilabel/prompts/single_page_answers.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(512, 512),
    )
]

config = Config(stages=stages)

