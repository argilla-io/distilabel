import os
from datasets import Dataset

from distilabel.tasks import EvolQualityTask
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline


# Create a sample dataset (this one is inspired from the distilabel-intel-orca-dpo-pairs)
sample_evol_quality = Dataset.from_dict(
    {
        'input': [
            'Please answer the following question: I want to test the ability of students to read a passage and answer questions about it. Could you please come up with a good question for the passage "In 1901, the Federation of Australia was the process by which the six separate British self-governing colonies of New South Wales, Queensland, South Australia, Tasmania, Victoria and Western Australia formed one nation. They kept the systems of government that they had developed as separate colonies but also would have a federal government that was responsible for matters concerning the whole nation. When the Constitution of Australia came into force, the colonies collectively became states of the Commonwealth of Australia."?\nAnswer:',
            "James runs a TV show and there are 5 main characters and 4 minor characters. He pays the minor characters $15,000 each episode. He paid the major characters three times as much. How much does he pay per episode? Let's be accurate as possible."
        ],
        'generation': [
            "Certainly! Here's a detailed and long answer to the question you provided:\n\nQuestion: Read the passage below and answer the following questions.\n\nPassage: In 1901, the Federation of Australia was the process by which the six separate British self-governing colonies of New South Wales, Queensland, South Australia, Tasmania, Victoria and Western Australia formed one nation. They kept the systems of government that they had developed as separate colonies but also would have a federal government that was responsible for matters concerning the whole nation. When the Constitution of Australia came into force, the colonies collectively became states of the Commonwealth of Australia.\n\n1. What event did the passage describe?\n2. How many colonies were involved in the Federation of Australia?\n3. What did the colonies keep after forming one nation?\n4. What was the purpose of the federal government established by the Constitution of Australia?\n5. What were the colonies called after they became states of the Commonwealth of Australia?\n\nAnswers:\n\n1. The passage describes the Federation of Australia, which was the process by which six separate British self-governing colonies formed one nation.\n2. Six colonies were involved in the Federation of Australia: New South Wales, Queensland, South Australia, Tasmania, Victoria, and Western Australia.\n3. The colonies kept the systems of government that they had developed as separate colonies.\n4. The federal government established by the Constitution of Australia was responsible for matters concerning the whole nation.\n5. After becoming states of the Commonwealth of Australia, the colonies were called states.\n\nIn-depth analysis:\n\nThe passage provides a brief overview of the Federation of Australia, which was the process by which six separate British self-governing colonies formed one nation in 1901. The colonies kept the systems of government that they had developed as separate colonies, but they also established a federal government that was responsible for matters concerning the whole nation. This federal government was established by the Constitution of Australia, which came into force on January 1, 1901.\n\nThe passage highlights the significance of the Federation of Australia by noting that it was a process by which the colonies collectively became states of the Commonwealth of Australia. This transformation marked a significant milestone in the history of Australia, as it created a unified nation out of six separate colonies.\n\nThe passage also provides information about the systems of government that were kept by the colonies after forming one nation. Specifically, it notes that the colonies kept the systems of government that they had developed as separate colonies. This suggests that the Federation of Australia did not result in a complete overhaul of the existing systems of government, but rather a reorganization of these systems to create a federal government that was responsible for matters concerning the whole nation.\n\nFinally, the passage uses descriptive language to emphasize the importance of the Constitution of Australia in the Federation of Australia. It notes that the Constitution came into force on January 1, 1901, and that it established the federal government that was responsible for matters concerning the whole nation. This language emphasizes the significance of the Constitution in the formation of the Federation of Australia and the creation of the Commonwealth of Australia.\n\nIn conclusion, the passage provides a detailed overview of the Federation of Australia, highlighting the significance of the Constitution of Australia in the formation of the Commonwealth of Australia. It also provides information about the systems of government that were kept by the colonies after forming one nation, and emphasizes the importance of the Constitution in the creation of the federal government responsible for matters concerning the whole nation.",
            'James pays the minor characters $15,000 each episode. Since there are 4 minor characters, he pays them a total of 4 * $15,000 = $60,000 per episode.\n\nThe major characters are paid three times as much. So, each major character gets paid 3 * $15,000 = $45,000 per episode.\n\nThere are 5 main characters, so he pays them a total of 5 * $45,000 = $225,000 per episode.\n\nIn total, James pays $225,000 (major characters) + $60,000 (minor characters) = $285,000 per episode.'
        ]
    }
)

# Create the pipeline
pipe = Pipeline(
    generator=OpenAILLM(
        task=EvolQualityTask(),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=1
    ),
)

# Run the pipeline in the sample dataset
sample_quality_dataset = pipe.generate(sample_evol_quality)

print(sample_quality_dataset.select_columns(["input", "generation", "generations"])[2])
# {
#   "input": "What happens next in this paragraph?\n\nShe then rubs a needle on a cotton ball then pushing it onto a pencil and wrapping thread around it. She then holds up a box of a product and then pouring several liquids into a bowl. she\nChoose your answer from: A. adds saucepan and shakes up the product in a grinder. B. pinches the thread to style a cigarette, and then walks away. C. then dips the needle in ink and using the pencil to draw a design on her leg, rubbing it off with a rag in the end. D. begins to style her hair and cuts it several times before parting the ends of it to show the hairstyle she has created.",
#   "generation": "C. She then dips the needle in ink and using the pencil to draw a design on her leg, rubbing it off with a rag in the end. In this option, she is continuing the process of using the needle, pencil, and thread, which is most related to what she was doing in the previous sentence.",
#   "generations": [
#     "C. Then, to everyone's surprise, she dips the needle in ink and starts using the pencil to draw an intricate design on her leg. The creativity in her actions is truly unparalleled. After showcasing her artwork, she skillfully rubs it off with a rag, leaving everyone in awe of her talent."
#   ]
# }