import os

from distilabel.llm import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask

endpoint_name = "aws-notus-7b-v1-4052" or os.getenv("HF_INFERENCE_ENDPOINT_NAME")
endpoint_namespace = "argilla" or os.getenv("HF_NAMESPACE")

pipe_generation = Pipeline(
    generator=InferenceEndpointsLLM(
        endpoint_name=endpoint_name,  # The name given of the deployed model
        endpoint_namespace=endpoint_namespace,  # This usually corresponds to the organization, in this case "argilla"
        token=os.getenv("HF_TOKEN"),  # hf_...
        task=TextGenerationTask(),
        max_new_tokens=512,
        do_sample=True,
        prompt_format="notus",
    ),
)
