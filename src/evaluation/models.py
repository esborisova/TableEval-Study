from abc import abstractmethod
from abc import ABC, abstractmethod
from typing import Dict, List
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModel(ABC):
    @abstractmethod
    def __call__(self, input: str) -> List:
        pass

    @abstractmethod
    def get_model_info(self) -> str:
        pass


class HFPipeModel(LanguageModel):
    def __init__(
        self,
        model_name: str,
        model_args: Dict,
        batch_size: int = 1,
        device: str = "cuda:0",
        random_seed: int = 0,
    ) -> None:
        super().__init__()

        set_seed(random_seed)
        self.model_name = model_name
        self.model_args = model_args
        self.device = device
        self.batch_size = batch_size
        if model_args:
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                device=self.device,
                trust_remote_code=True,
                model_kwargs=self.model_args,
                output_scores=True,
            )
        else:
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                device=self.device,
                trust_remote_code=True,
                output_scores=True,
            )

    def __call__(self, input: List, **kwargs):
        return self.pipe(
            input, batch_size=self.batch_size, output_scores=True, **kwargs
        )

    def get_model_info(self):
        return f"HF Model: {self.model_name}"


class HFModel(LanguageModel):
    def __init__(
        self,
        model_name: str,
        model_args: Dict,
        batch_size: int = 1,
        device: str = "cuda:0",
        random_seed: int = 0,
    ) -> None:
        super().__init__()

        set_seed(random_seed)
        self.model_name = model_name
        self.model_args = model_args
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, **self.model_args
        )

    def __call__(self, prompts: List, **kwargs):
        inputs = self.tokenizer(prompts, truncation=True, padding=True, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )
        generated_text = [
            self.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in output.sequences.cpu()
        ]  # skip_special_token = true
        # logits shape: (batch_size, vocab_size)
        logits = output.scores
        return generated_text, logits

    def get_model_info(self):
        return f"HF Model: {self.model_name}"
