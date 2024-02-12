import os
from datasets import Dataset

from distilabel.tasks import ComplexityScorerTask
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline


# Create a sample dataset (this one is inspired from the distilabel-sample-evol-complexity dataset)
sample_evol_complexity = Dataset.from_dict(
    {
        'generations': [
            [
                'Generate a catchy tagline for a new high-end clothing brand\n',
                "Devise a captivating and thought-provoking tagline that effectively represents the unique essence and luxurious nature of an upcoming luxury fashion label. Additionally, ensure that the tagline encapsulates the brand's core values and resonates with the discerning tastes of its exclusive clientele."
            ],
            [
                'How can I create a healthier lifestyle for myself?\n',
                'What are some innovative ways to optimize physical and mental wellness while incorporating sustainable practices into daily routines?'
            ]
        ]
    }
)

# Create the pipeline
pipe = Pipeline(
    labeller=OpenAILLM(
        task=ComplexityScorerTask(),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1
    )
)

# Run the pipeline in the sample dataset
new_dataset = pipe.generate(sample_evol_complexity.select(range(3,5)))

print(new_dataset.select_columns(["generations", "rating"])[:])
# {
#   "generations": [
#     [
#       "Generate a catchy tagline for a new high-end clothing brand\n",
#       "Devise a captivating and thought-provoking tagline that effectively represents the unique essence and luxurious nature of an upcoming luxury fashion label. Additionally, ensure that the tagline encapsulates the brand's core values and resonates with the discerning tastes of its exclusive clientele."
#     ],
#     [
#       "How can I create a healthier lifestyle for myself?\n",
#       "What are some innovative ways to optimize physical and mental wellness while incorporating sustainable practices into daily routines?"
#     ]
#   ],
#   "rating": [
#     [1.0, 3.0],
#     [1.0, 2.0]
#   ]
# }