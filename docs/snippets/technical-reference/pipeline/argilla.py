import argilla as rg
from argilla.client.feedback.integrations.sentencetransformers import (
    SentenceTransformersExtractor,
)
from argilla.client.feedback.integrations.textdescriptives import (
    TextDescriptivesExtractor,
)

rg.init(api_key="<YOUR_ARGILLA_API_KEY>", api_url="<YOUR_ARGILLA_API_URL>")

rg_dataset = pipe_dataset.to_argilla()
rg_dataset.push_to_argilla(name="preference-dataset", workspace="admin")

# with a custom `vector_strategy``
vector_strategy = SentenceTransformersExtractor(model="TaylorAI/bge-micro-v2")
rg_dataset = pipe_dataset.to_argilla(vector_strategy=vector_strategy)

# with a custom `metric_strategy`
metric_strategy = TextDescriptivesExtractor(model="en_core_web_sm")
rg_dataset = pipe_dataset.to_argilla(metric_strategy=vector_strategy)