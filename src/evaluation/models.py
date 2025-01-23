from abc import abstractmethod
from abc import ABC, abstractmethod
from typing import Dict, List
from transformers import pipeline, set_seed
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
)


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
        if not self.use_chat_template:
            inputs = self.processor(
                prompts, truncation=True, padding=True, return_tensors="pt"
            ).to(self.device)
        else:
            # TODO:
            inputs = [
                self.processor.apply_chat_template(
                    prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self.device)
                for prompt in prompts
            ]
        output = self.step(inputs=inputs, **kwargs)

        decoded_outputs = []
        for i in range(inputs.input_ids.shape[0]):  # Loop over batch samples
            input_length = inputs.input_ids.shape[1] 
            generated_token_ids = output.sequences[i, input_length:].cpu()  # Skip input tokens
            decoded_text = self.processor.decode(generated_token_ids.tolist(), skip_special_tokens=True)
            decoded_outputs.append(decoded_text)
        logits = output.scores
        return decoded_outputs, logits

    def multi_modal_forward(self, mm_input: List[tuple], **kwargs):
        images = [i[0] for i in mm_input]
        # adding the <image> special_token to the prompt at the beginning
        if not self.use_chat_template:
            texts = [self.image_token + i[1] for i in mm_input]
            inputs = self.processor(
                images=images,
                text=texts,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            texts = [i[1] for i in mm_input]
            template_inputs = [
                self.processor.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                for prompt in texts
            ]
            inputs = self.processor(
                images=images,
                text=template_inputs,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        output = self.step(inputs, **kwargs)
        #    generated_text = self.processor.batch_decode(
        decoded_outputs = []
        for i in range(inputs.input_ids.shape[0]):  # Loop over batch samples
            input_length = inputs.input_ids.shape[1] 
            generated_token_ids = output.sequences[i, input_length:].cpu()  # Skip input tokens
            decoded_text = self.processor.tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
            decoded_outputs.append(decoded_text)
        logits = output.scores
        return decoded_outputs, logits

    def step(self, inputs, **kwargs):
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs,
            )
        return output

    def __call__(self, input, **kwargs):
        if not self.multi_modal:
            return self.forward(input, **kwargs)
        else:
            return self.multi_modal_forward(input, **kwargs)

    def get_model_info(self):
        return f"{self.model_name}"


# WARNING: old model class
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
        return f"{self.model_name}"


# TODO: Ovis not working yet
class OvisModel(LanguageModel):
    def __init__(
        self,
        model_name: str,
        model_args: Dict,
        batch_size: int = 1,
        device: str = "cuda:0",
        random_seed: int = 0,
        multi_modal: bool = True,
        special_token_for_image: str = "<image>",
    ) -> None:
        super().__init__()

        # Ovis is a multi_modal LLM but has to be loaded with AutoModelForCausalLM
        set_seed(random_seed)
        self.model_name = model_name
        self.model_args = model_args
        self.device = device
        self.batch_size = batch_size
        self.image_token = special_token_for_image
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            **self.model_args,
        )
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def __call__(self, mm_input: List, **kwargs):
        images = [[i[0]] for i in mm_input]
        # adding the <image> special_token to the prompt at the beginning
        texts = [self.image_token + i[1] for i in mm_input]
        # format conversation
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(texts, images)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        pixel_values = [
            pixel_values.to(
                dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device
            )
        ]

        # generate output
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **kwargs,
            )[0]
        output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output

    def get_model_info(self):
        return f"{self.model_name}"
