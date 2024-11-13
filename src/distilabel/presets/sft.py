import os
from typing import Optional
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns
from distilabel.steps.tasks import MagpieGenerator
from distilabel.llms import InferenceEndpointsLLM

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = "You are a customer support agent for a phone company. \
    Your purpose is to assist customers with their phone-related issues, \
    but you are not very patient and tend to be a bit rude. User queries  \
    will be straightforward and clear, but you will respond in a somewhat \
    blunt and curt manner. Remember to keep your responses concise and to \
    the point. User queries are often about phone plans, billing, and \
    technical issues. Your responses should be direct and focus on resolving \
    the issue at hand, but with a slightly abrasive tone. User queries will be \
    concise and to the point, User queries are often about phone plans, billing, \
    and technical issues."


class SFTPipeline:

    def __init__(self, hf_token=None) -> None:
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        with Pipeline(name="sft") as pipeline:
            magpie = MagpieGenerator(
                llm=InferenceEndpointsLLM(
                    model_id=MODEL,
                    tokenizer_id=MODEL,
                    magpie_pre_query_template="llama3",
                    generation_kwargs={
                        "temperature": 0.9,
                        "do_sample": True,
                        "max_new_tokens": 2048,
                        "stop_sequences": [
                            "<|eot_id|>",
                            "<|start_header_id|>",
                            "assistant",
                            " \n\n",
                        ],
                    },
                    api_key=hf_token,
                ),
                n_turns=1,
                num_rows=10,
                batch_size=1,
                system_prompt=SYSTEM_PROMPT,
                output_mappings={"instruction": "prompt", "response": "completion"},
            )
            keep_columns = KeepColumns(
                columns=["prompt", "completion"] + ["model_name"],
            )
            magpie.connect(keep_columns)

        self.pipeline = pipeline

    def run(self):
        return self.pipeline.run()
