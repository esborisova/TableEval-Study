from abc import abstractmethod
from abc import ABC, abstractmethod
from typing import Dict, List
from transformers import pipeline, set_seed
import torch.nn.functional as F
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
)
from accelerate import infer_auto_device_map
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from litellm import batch_completion


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
        device: str = "cuda:0",
        random_seed: int = 0,
        multi_modal: bool = False,
        special_token_for_image: str = "<image>",
        use_chat_template: bool = False,
        accelerate: bool = False,
    ) -> None:
        super().__init__()

        set_seed(random_seed)
        self.model_name = model_name
        self.model_args = model_args
        self.device = device
        self.batch_size = batch_size
        self.multi_modal = multi_modal
        self.image_token = special_token_for_image
        self.use_chat_template = use_chat_template
        self.processor = self.load_processor()
        self.model = self.load_model()
        if not multi_modal:
            if self.processor.pad_token is None:
                self.processor.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.processor))
        if accelerate:
            num_devices = int(device.split(":")[-1]) + 1
            max_memory = {i: "16GiB" for i in range(0, num_devices)}
            # Automatically map layers to multiple GPUs
            device_map = infer_auto_device_map(self.model, max_memory=max_memory)

            # Load model weights and distribute across GPUs
            self.model = load_checkpoint_and_dispatch(
                self.model, model_name, device_map=device_map, offload_folder="offload"
            )

            # Move model to appropriate device
            self.model.to("cuda")

    def __call__(self, input, **kwargs):
        return self.forward(input, **kwargs)

    def get_model_info(self):
        return f"{self.model_name}"

    def load_model(self):
        if not self.multi_modal:
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                **self.model_args,
            )
        else:
            return AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                device_map=self.device,
                **self.model_args,
            )

    def load_processor(self):
        if not self.multi_modal:
            return AutoTokenizer.from_pretrained(self.model_name)
        else:
            return AutoProcessor.from_pretrained(self.model_name)

    def forward(self, prompts: List, **kwargs):
        inputs = self.generate_inputs(prompts)

        output = self.step(inputs=inputs, **kwargs)

        decoded_outputs, logits = self.decode_outputs(output, inputs)

        return decoded_outputs, logits

    def step(self, inputs, **kwargs):
        self.model.eval()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs,
            )
        return output

    def decode_outputs(self, output, inputs):
        decoded_outputs = []
        for i in range(inputs.input_ids.shape[0]):  # Loop over batch samples
            input_length = inputs.input_ids.shape[1]
            generated_token_ids = output.sequences[
                i, input_length:
            ].cpu()  # Skip input tokens
            # decode the generated part
            if self.multi_modal:
                decoded_text = self.processor.tokenizer.decode(
                    generated_token_ids.tolist(), skip_special_tokens=True
                )
            else:
                decoded_text = self.processor.decode(
                    generated_token_ids.tolist(), skip_special_tokens=True
                )
            decoded_outputs.append(decoded_text)
            logits = output.scores
        return decoded_outputs, logits

    def generate_inputs(self, raw_input):
        if self.multi_modal:
            if not self.use_chat_template:
                texts = [self.image_token + i[1] for i in raw_input]
            else:
                text_parts = [i[1] for i in raw_input]
                texts = [
                    self.processor.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    for prompt in text_parts
                ]
            images = [i[0] for i in raw_input]
            inputs = self.processor(
                images=images,
                text=texts,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            if self.use_chat_template:
                raw_input = [
                    self.processor.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for prompt in raw_input
                ]
            inputs = self.processor(
                raw_input,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        return inputs


class LiteLLM(LanguageModel):
    def __init__(
        self,
        model_name: str,
        model_args: Dict,
        random_seed: int = 0,
        multi_modal: bool = False,
        use_chat_template: bool = False,
        api_key: str = "",
    ) -> None:
        super().__init__()

        set_seed(random_seed)
        self.model_name = model_name
        self.model_args = model_args
        self.multi_modal = multi_modal
        self.use_chat_template = use_chat_template
        if api_key:
            name = model_name.split("/")[0].upper()
            os.environ[f"{name}_API_KEY"] = api_key

    def __call__(self, input, **kwargs):
        return self.forward(input, **kwargs)

    def get_model_info(self):
        return f"{self.model_name}"

    def forward(self, prompts: List, **kwargs):
        if not self.use_chat_template:
            prompts = self.generate_inputs(prompts)

        output = self.step(inputs=prompts, **kwargs, **self.model_args)

        return output, None

    def step(self, inputs, **kwargs):
        output = batch_completion(
            model=self.model_name, messages=inputs, **kwargs, **self.model_args
        )
        return [o["choices"][0]["message"]["content"] for o in output]

    def generate_inputs(self, raw_input):
        if self.multi_modal:
            # TODO: base64 convertion
            return [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": i[1]},
                            {"type": "image_url", "image_url": i[0]},
                        ],
                    }
                ]
                for i in raw_input
            ]
        else:
            return [
                [
                    {
                        "role": "user",
                        "content": i,
                    }
                ]
                for i in raw_input
            ]
