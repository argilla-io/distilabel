from typing import List

from huggingface_hub import InferenceClient

class LLM:
    def __init__(self, model, tokenizer,num_responses=2, temperature=1.0, max_length=256,
                 top_k=50, top_p=0.95, repetition_penalty=1.2, no_repeat_ngram_size=2, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.num_responses = num_responses
        self.temperature = temperature
        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.extra_args = kwargs  # For any extra arguments
    
    def _validate_input(self, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text)}")
        
    # TODO:Exclude input text from outputs
    def generate_responses(self, text: str) -> List[str]:
        self._validate_input(text)
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=self.max_length,
            num_return_sequences=self.num_responses,
            num_beams=max(2, self.num_responses),
            temperature=self.temperature,
            do_sample=True,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return responses

class LLMInferenceEndpoint:
    def __init__(self, client: InferenceClient, num_responses=2, system_prompt=None, base_prompt=None, temperature=1.0, max_length=256, max_new_tokens=512,
                 top_k=50, top_p=0.95, repetition_penalty=1.2, no_repeat_ngram_size=2, **kwargs):
        # TODO: Move this to a right place
        default_system_prompt = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
            " while being safe. Your answers should not include any harmful, unethical, racist, sexist,"
            " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased"
            " and positive in nature.\nIf a question does not make any sense, or is not factually coherent,"
            " explain why instead of answering something not correct. If you don't know the answer to a"
            " question, please don't share false information."
        )
        default_base_prompt = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        self.num_responses = num_responses
        self.client = client
        self.temperature = temperature
        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.extra_args = kwargs  # For any extra arguments
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt or default_system_prompt
        self.base_prompt = base_prompt or default_base_prompt

    def _validate_input(self, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text)}")
        
    def generate_responses(self, text: str) -> List[str]:
        self._validate_input(text)
        prompt = self.base_prompt.format(system_prompt=self.system_prompt, prompt=text)
        responses = [
            self.client.text_generation(
                prompt, details=True, max_new_tokens=self.max_new_tokens, top_k=self.top_k, top_p=self.top_p,
                temperature=self.temperature, repetition_penalty=self.repetition_penalty, stop_sequences=["</s>"],
            ).generated_text
            for i in range(self.num_responses)
        ]
        return responses
