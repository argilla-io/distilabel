import os
import openai

class RatingModelConfig:
    def __init__(self, model="gpt-4", num_responses=2, max_tokens=150, top_p=0.6, presence_penalty=0, **kwargs):
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.num_responses = num_responses
        self.extra_args = kwargs  

class RatingModel:
    def __init__(self, config=None, rating_prompt=None, openai_api_key=None):
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if self.openai_api_key is None:
            raise ValueError("The OpenAI API key must be provided either as an argument or as the OPENAI_API_KEY environment variable.")

        self.config = config or RatingModelConfig()
        self.rating_prompt = rating_prompt or RatingPrompt()
        self.system_prompt = self.rating_prompt.system_prompt

    def rate_responses(self, response_texts, input_text):
        user_prompt = self.rating_prompt.user_prompt.format(
            text_sections_annotation="\n".join(f"<text {i + 1}> {text}" for i, text in enumerate(response_texts)),
            instruction=input_text,
        )
        print(user_prompt)
        openai.api_key = self.openai_api_key
        try:
            response = openai.ChatCompletion.create(
                model=self.config.model, 
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                presence_penalty=self.config.presence_penalty,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate rating: {e}")
        
        # Extract the rating and rationale from the response
        rating_output = response['choices'][0]['message']["content"].strip()
        
        sections = rating_output.split("#### Output for Text ")[1:]  # Ignore any content before the first header
        parsed_output = []
        for section in sections:
            header, rating_line, rationale_line = section.strip().split('\n', 2)
            text_num = int(header.split()[-1])  # This is the number following "Output for Text "
            rating = rating_line.split(": ")[1]
            rationale = rationale_line.split(": ")[1]
            parsed_output.append({"rating": rating, "rationale": rationale})

        return {"rated_responses": parsed_output}
    
class RatingPrompt:

    default_system_prompt = "Your role is to evaluate text quality based on given criteria."

    default_template = """
    # Informativeness / Helpfulness Assessment

    Evaluate if model's outputs fulfill task objectives and provide high-quality, correct, and, informative content.

    Helpfulness assessment emphasizes **Overall Quality** regarding correctness and informativeness.

    **Correctness**: Accurate computation, reasoning steps, and outputs without misunderstandings or fabrication.

    Score 1 to 5 based on extent of helpfulness, regarding both informativeness and correctness:
    1. **Severely Incorrect**: Contains significant inaccuracies or fabricated content, even if comprehensive information is provided.
    2. **Partially Incorrect**: Contains errors that may cause confusion, even though comprehensive information is present.
    3. **Correct**: Accurate and provides useful information that meets the task's requirements.
    4. **Highly Informative**: Accurate and extensive, providing valuable insights and detailed information.
    5. **Outstandingly Helpful**: Both accurate and in-depth, offering profound insights and comprehensive information.

    ---

    ## Format

    ### Input
    Instruction: [Specify task goal and restrictions]

    Texts:
    {text_sections_input}

    ### Output
    {text_sections_output}

    ---

    ## Annotation

    ### Input
    Instruction: {{instruction}}

    Texts:
    {{text_sections_annotation}}

    ### Output
    """

    def __init__(self, num_responses=2, template=None, system_prompt=None):
        self.num_responses = num_responses
        self.template = template or self.default_template
        self.system_prompt = system_prompt or self.default_system_prompt
        self.user_prompt = self._generate_final_prompt()  # Generate the final template during initialization

    def generate_text_sections(self):
        text_sections_input = "\n".join(f"<text {i + 1}> [Text {i + 1}]" for i in range(self.num_responses))
        text_sections_output = "\n\n".join(f"""
    #### Output for Text {i + 1}
    Rating: [Rating for text {i + 1}]
    Rationale: [Rationale for the rating in short sentences]
    """ for i in range(self.num_responses))
        return text_sections_input, text_sections_output
    
    def _generate_final_prompt(self):
        text_sections_input, text_sections_output = self.generate_text_sections()
        return self.template.format(text_sections_input=text_sections_input, text_sections_output=text_sections_output)