# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

from datasets import load_dataset

from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import FormatTextGenerationSFT, LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration


class SocialAI(TextGeneration):
    follower_type: Literal["supporter", "troll", "alarmist"] = "supporter"
    system_prompt: str = (
        "You are an AI assistant expert at simulating user interactions. "
        "You must answer as if you were a '{follower_type}', be concise answer with no more than 200 characters, nothing else."
        "Here are some traits to use for your personality:\n\n"
        "{traits}"
    )
    template: str = "You are the folowing persona:\n\n{{ persona }}\n\nWhat would you say to the following?\n\n {{ post }}"
    columns: str | list[str] = ["persona", "post"]

    _follower_traits: dict[str, str] = {
        "supporter": (
            "- Encouraging and positive\n"
            "- Tends to prioritize enjoyment and relaxation\n"
            "- Focuses on the present moment and short-term pleasure\n"
            "- Often uses humor and playful language\n"
            "- Wants to help others feel good and have fun\n"
        ),
        "troll": (
            "- Provocative and confrontational\n"
            "- Enjoys stirring up controversy and conflict\n"
            "- Often uses sarcasm, irony, and mocking language\n"
            "- Tends to belittle or dismiss others' opinions and feelings\n"
            "- Seeks to get a rise out of others and create drama\n"
        ),
        "alarmist": (
            "- Anxious and warning-oriented\n"
            "- Focuses on potential risks and negative consequences\n"
            "- Often uses dramatic or sensational language\n"
            "- Tends to be serious and stern in tone\n"
            "- Seeks to alert others to potential dangers and protect them from harm (even if it's excessive or unwarranted)\n"
        ),
    }

    def load(self) -> None:
        super().load()
        self.system_prompt = self.system_prompt.format(
            follower_type=self.follower_type,
            traits=self._follower_traits[self.follower_type],
        )


posts = [
    {
        "post": "Hmm, ok now I'm torn: should I go for healthy chicken tacos or unhealthy beef tacos for late night cravings?"
    },
    {
        "post": "I need to develop a training course for my company on communication skills. Need to decide how deliver it remotely."
    },
    {
        "post": "I'm always 10 minutes late to meetups but no one's complained. Could this be annoying to them?"
    },
]

personas = (
    load_dataset("argilla/FinePersonas-v0.1-clustering-100k", split="train")
    .shuffle()
    .select(range(3))
    .select_columns("persona")
    .to_list()
)

data = []
for post in posts:
    for persona in personas:
        data.append({"post": post["post"], "persona": persona["persona"]})


with Pipeline(name="Social AI Personas") as pipeline:
    loader = LoadDataFromDicts(data=data, batch_size=1)

    llm = InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        generation_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 256,
        },
    )

    for follower_type in ["supporter", "troll", "alarmist"]:
        follower = SocialAI(
            llm=llm,
            follower_type=follower_type,
            name=f"{follower_type}_user",
            output_mappings={"generation": f"interaction_{follower_type}"},
        )
        format_sft = FormatTextGenerationSFT(
            name=f"format_sft_{follower_type}",
            input_mappings={
                "instruction": "post",
                "generation": f"interaction_{follower_type}",
            },
        )
        loader >> follower >> format_sft


if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    distiset.push_to_hub("plaguss/FinePersonas-SocialAI-test", include_script=True)
