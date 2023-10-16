from typing import List

class LLM:
    def __init__(self, model, tokenizer,num_responses=2, temperature=1.0, max_length=1024,
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
