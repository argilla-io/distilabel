import os

from distilabel.llm import InferenceEndpointsLLM
from distilabel.tasks import TextGenerationTask

endpoint_name = "aws-notus-7b-v1-4052" or os.getenv("HF_INFERENCE_ENDPOINT_NAME")
endpoint_namespace = "argilla" or os.getenv("HF_NAMESPACE")
token = os.getenv("HF_TOKEN")  # hf_...

llm = InferenceEndpointsLLM(
    endpoint_name=endpoint_name,
    endpoint_namespace=endpoint_namespace,
    token=token,
    task=TextGenerationTask(),
    max_new_tokens=512,
    prompt_format="notus",
)
result = llm.generate([{"input": "What are critique LLMs?"}])
# print(result[0][0]["parsed_output"]["generations"])
# Critique LLMs (Long Land Moore Machines) are artificial intelligence models designed specifically for analyzing and evaluating the quality or worth of a particular subject or object. These models can be trained on a large dataset of reviews, ratings, or commentary related to a product, service, artwork, or any other topic of interest.
# The training data can include both positive and negative feedback, helping the LLM to understand the nuanced aspects of quality and value. The model uses natural language processing (NLP) techniques to extract meaningful insights, including sentiment analysis, entity recognition, and text classification.
# Once the model is trained, it can be used to analyze new input data and provide a critical assessment based on its learned understanding of quality and value. For example, a critique LLM for movies could evaluate a new film and generate a detailed review highlighting its strengths, weaknesses, and overall rating.
# Critique LLMs are becoming increasingly useful in various industries, such as e-commerce, education, and entertainment, where they can provide objective and reliable feedback to help guide decision-making processes. They can also aid in content optimization by highlighting areas of improvement or recommending strategies for enhancing user engagement.
# In summary, critique LLMs are powerful tools for analyzing and evaluating the quality or worth of different subjects or objects, helping individuals and organizations make informed decisions with confidence.
