from textwrap import dedent

from distilabel.tasks.preference.ultrafeedback import Rating, UltraFeedbackTask

task_description = dedent(
    """
    # General Text Quality Assessment
    Evaluate the model's outputs based on various criteria:
    1. **Correctness & Informativeness**: Does the output provide accurate and helpful information?
    2. **Honesty & Uncertainty**: How confidently does the model convey its information, and does it express uncertainty appropriately?
    3. **Truthfulness & Hallucination**: Does the model introduce misleading or fabricated details?
    4. **Instruction Following**: Does the model's output align with given instructions and the user's intent?
    Your role is to provide a holistic assessment considering all the above factors.

    **Scoring**: Rate outputs 1 to 3 based on the overall quality, considering all aspects:
    """
)

ratings = [
    Rating(value=1, description="Low Quality"),
    Rating(value=2, description="Moderate Quality"),
    Rating(value=3, description="Good Quality"),
]

ultrafeedback_task = UltraFeedbackTask(
    system_prompt="Your role is to evaluate text quality based on given criteria",
    task_description=task_description,
    ratings=ratings,
)
