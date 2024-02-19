import os

from datasets import Dataset
from distilabel.tasks import QualityScorerTask
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline


# Create a sample dataset (this one is inspired from the distilabel-intel-orca-dpo-pairs)
sample_evol_quality = Dataset.from_dict(
    {
        'input': [
            'Please answer the following question: I want to test the ability of students to read a passage and answer questions about it. Could you please come up with a good question for the passage "In 1901, the Federation of Australia was the process by which the six separate British self-governing colonies of New South Wales, Queensland, South Australia, Tasmania, Victoria and Western Australia formed one nation. They kept the systems of government that they had developed as separate colonies but also would have a federal government that was responsible for matters concerning the whole nation. When the Constitution of Australia came into force, the colonies collectively became states of the Commonwealth of Australia."?\nAnswer:',
            "James runs a TV show and there are 5 main characters and 4 minor characters. He pays the minor characters $15,000 each episode. He paid the major characters three times as much. How much does he pay per episode? Let's be accurate as possible."
        ],
        'generations': [
            [
                " Certainly! Here's a detailed and long answer to the question you provided:\n\nQuestion: Read the passage below and answer the following questions.\n\nPassage: In 1901, the Federation of Australia was the process by which the six separate British self-governing colonies of New South Wales, Queensland, South Australia, Tasmania, Victoria and Western Australia formed one nation. They kept the systems of government that they had developed as separate colonies but also would have a federal government that was responsible for matters concerning the whole nation. When the Constitution of Australia came into force, the colonies collectively became states of the Commonwealth of Australia.\n\n1. What event did the passage describe?\n2. How many colonies were involved in the Federation of Australia?\n3. What did the colonies keep after forming one nation?\n4. What was the purpose of the federal government established by the Constitution of Australia?\n5. What were the colonies called after they became states of the Commonwealth of Australia?\n\nAnswers:\n\n1. The passage describes the Federation of Australia, which was the process by which six separate British self-governing colonies formed one nation.\n2. Six colonies were involved in the Federation of Australia: New South Wales, Queensland, South Australia, Tasmania, Victoria, and Western Australia.\n3. The colonies kept the systems of government that they had developed as separate colonies.\n4. The federal government established by the Constitution of Australia was responsible for matters concerning the whole nation.\n5. After becoming states of the Commonwealth of Australia, the colonies were called states.\n\nIn-depth analysis:\n\nThe passage provides a brief overview of the Federation of Australia, which was the process by which six separate British self-governing colonies formed one nation in 1901. The colonies kept the systems of government that they had developed as separate colonies, but they also established a federal government that was responsible for matters concerning the whole nation. This federal government was established by the Constitution of Australia, which came into force on January 1, 1901.\n\nThe passage highlights the significance of the Federation of Australia by noting that it was a process by which the colonies collectively became states of the Commonwealth of Australia. This transformation marked a significant milestone in the history of Australia, as it created a unified nation out of six separate colonies.\n\nThe passage also provides information about the systems of government that were kept by the colonies after forming one nation. Specifically, it notes that the colonies kept the systems of government that they had developed as separate colonies. This suggests that the Federation of Australia did not result in a complete overhaul of the existing systems of government, but rather a reorganization of these systems to create a federal government that was responsible for matters concerning the whole nation.\n\nFinally, the passage uses descriptive language to emphasize the importance of the Constitution of Australia in the Federation of Australia. It notes that the Constitution came into force on January 1, 1901, and that it established the federal government that was responsible for matters concerning the whole nation. This language emphasizes the significance of the Constitution in the formation of the Federation of Australia and the creation of the Commonwealth of Australia.\n\nIn conclusion, the passage provides a detailed overview of the Federation of Australia, highlighting the significance of the Constitution of Australia in the formation of the Commonwealth of Australia. It also provides information about the systems of government that were kept by the colonies after forming one nation, and emphasizes the importance of the Constitution in the creation of the federal government responsible for matters concerning the whole nation.",
                "Certainly! Here's a more detailed answer to the question you provided with additional analysis:\n\nQuestion: Read the passage below and answer the following questions.\n\nPassage: In 1901, the Federation of Australia was the process by which the six separate British self-governing colonies of New South Wales, Queensland, South Australia, Tasmania, Victoria and Western Australia formed one nation. They kept the systems of government that they had developed as separate colonies but also would have a federal government that was responsible for matters concerning the whole nation. When the Constitution of Australia came into force, the colonies collectively became states of the Commonwealth of Australia.\n\n1. What"
            ],
            [
                'James pays the minor characters $15,000 each episode. Since there are 4 minor characters, he pays them a total of 4 * $15,000 = $60,000 per episode.\n\nThe major characters are paid three times as much. So, each major character gets paid 3 * $15,000 = $45,000 per episode.\n\nThere are 5 main characters, so he pays them a total of 5 * $45,000 = $225,000 per episode.\n\nIn total, James pays $225,000 (major characters) + $60,000 (minor characters) = $285,000 per episode.', 
                "In James' TV show, he pays each of the 4 minor characters $15,000 per episode, totaling $60,000. The major characters, being paid three times as much, receive $45,000 each per episode. With 5 main characters, James pays a total of $225,000 for them. Therefore, the total payment per episode is $285,000, consisting of $225,000 for the major characters and $60,000 for the minor characters."
            ]
        ]
   }
)

# Create the pipeline to label the dataset with theQualityScorerTask
pipe_labeller = Pipeline(
    labeller=OpenAILLM(
        task=QualityScorerTask(),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
        max_new_tokens=1024
    )
)

# Run the pipeline to get the scoring for the datase
quality_labelled_dataset = pipe_labeller.generate(sample_evol_quality)
print(quality_labelled_dataset.select_columns(["labelling_prompt", "rating"])[0])
# {
#     'labelling_prompt': [
#         {'content': '', 'role': 'system'},
#         {
#             'content': 'Rank the following responses provided by different AI assistants to the user’s question\naccording to the quality of their response. Score each response from 1 to 2, with 3\nreserved for responses that are already very well written and cannot be improved further.\nYour evaluation should consider factors such as helpfulness, relevance, accuracy, depth,\ncreativity, and level of detail of the response.\nUse the following format:\n[Response 1] Score:\n[Response 2] Score:\n...\n#Question#: You will be given a definition of a task first, then some input of the task.\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\n\nAFC Ajax (amateurs)\'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\nOutput:\n#Response List#:\n\n[Response 1] [\n  ["AFC Ajax (amateurs)", "has ground", "Sportpark De Toekomst"],\n  ["Ajax Youth Academy", "plays at", "Sportpark De Toekomst"]\n]\n[Response 2] The RDF triplets generated from the input sentence "AFC Ajax (amateurs)\'s ground is Sportpark De Toekomst where Ajax Youth Academy also play" accurately capture the relationships present. The output is a list of triplets that includes ["AFC Ajax (amateurs)", "has ground", "Sportpark De Toekomst"] and ["Ajax Youth Academy", "plays at", "Sportpark De Toekomst"]. These triplets represent the structure and semantics of the sentence.',
#             'role': 'user'
#         }
#     ],
#     'rating': [2.0, 3.0]
# }