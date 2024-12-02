from abc import abstractmethod
from abc import ABC, abstractmethod
from typing import Dict, List
from transformers import pipeline, set_seed


class LanguageModel(ABC):
    @abstractmethod
    def __call__(self, input: str) -> List:
        pass

    @abstractmethod
    def get_model_info(self) -> str:
        pass


class HFModel(LanguageModel):
    def __init__(
        self,
        model_name: str,
        model_args: Dict,
        batch_size: int = 1,
        device: str = "cuda",
        random_seed: int = 0,
        numpy_random_seed: int = 1234,
        torch_random_seed: int = 1234,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_args = model_args
        self.devide = device
        self.batch_size = batch_size
        set_seed(random_seed)
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_args,
        )

    def __call__(self, input: str, **kwargs):
        return self.pipe(input, batch_size=self.batch_size, **kwargs)[0][
            "generated_text"
        ]

    def get_model_info(self):
        return f"HF Model: {self.model_name}"
