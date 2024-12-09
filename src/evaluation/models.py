from abc import abstractmethod
from abc import ABC, abstractmethod
from typing import Dict, List
from transformers import pipeline, set_seed
import torch


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
            )
        else:
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                device=self.device,
                trust_remote_code=True,
            )

    def __call__(self, input: List, **kwargs):
        return self.pipe(input, batch_size=self.batch_size, **kwargs)[0]

    def get_model_info(self):
        return f"HF Model: {self.model_name}"
