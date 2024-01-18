from distilabel.tasks import SelfInstructTask

system_prompt: str = "You are an expert Haiku writer, writing the best and most diverse Haikus given topics as inputs."

application_description = (
    "An AI assistant adept at writing Haiku. "
    "It expects complete suggestions from users providing details of the kind of haiku they want. "
    "The AI assistant will help users write haiku about particular topics and is willing to accept requests related to a specific subject or object or a more abstract request"
    "based on an emotion, theme or vibe."
)

criteria_queries = (
    "Incorporate a diverse range of verbs, avoiding repetition. "
    "Ensure queries are compatible with AI model's text generation functions and are limited to 1-2 sentences."
    "Design queries to be self-contained and standalone."
)

instruction_task = SelfInstructTask(
    num_instructions=15,
    application_description=application_description,
    criteria_for_query_generation=criteria_queries,
)
