from distilabel.tasks import UltraJudgeTask

task = UltraJudgeTask()

input = (
    "Can you provide a corrected version of the following sentence using proper "
    'English grammar? "We going to the beach" Additionally, could you please '
    "provide your correction in an Excel table format with the following columns: "
    "| Incorrect Sentence | Corrected Sentence | |-------------------|--------------------|"
)

generations = [
    (
        "| Incorrect Sentence | Corrected Sentence |\n|-------------------|-------------------"
        '-----|\n| "We going to the beach" | "We are going to the beach" |\n\nCorrectio'
        'n: The verb in the second sentence ("are") changes to reflect the subject\'s ("w'
        'e") agreement with the verb "be." This is called subject-verb agreement. In the '
        'first sentence, the verb "going" infers that the action is ongoing or in a contin'
        "uous state, which is not the case. Therefore, the second sentence is grammatically "
        "correct."
    ),
    (
        "| Incorrect Sentence | Corrected Sentence |\n|-------------------|-------------------"
        "-----|\n| We going to the beach | We are going to the beach | \n\nHere's a breakdo"
        'wn of the correction:\n\n- "We going to the beach" is an example of a subject-ve'
        'rb agreement error. The verb changing from the third person singular ("is") to t'
        'he third person plural ("are") in this instance, as there are multiple people go'
        'ing to the beach.\n- The "g" in "going" changes to an "e" due to a hard "g"'
        ' sound being followed by an "e," which is a common spelling rule in English.'
    ),
]


prompt = task.generate_prompt(input, generations)
print(prompt.format_as("default"))  # format as "openai", "zephyr", "llama", ...
