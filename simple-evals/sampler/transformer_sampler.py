import time
from typing import Any
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from ..types import MessageList, SamplerBase, SamplerResponse


class TransformerSampler(SamplerBase):
    """
    Sample from Transformer models
    """

    def __init__(
        self,
        model_name: str = "EleutherAI/gpt-neo-1.3B",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Ensure the model is in evaluation mode
        self.model.eval()

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        print("Message list:", message_list)
        while True:
            try:
                if self.tokenizer.chat_template:
                    prompt_tokens = self.tokenizer.apply_chat_template(
                        message_list,
                        tokenize=True,
                        return_tensors="pt",
                        add_generation_prompt=True  # Important for chat models
                    )
                    # Move to the same device as the model if necessary
                    inputs = {"input_ids": prompt_tokens.to(self.model.device)}
                    # Attention mask is automatically created by apply_chat_template when tokenize=True
                    # and return_tensors="pt"
                    # inputs["attention_mask"] = self.tokenizer(prompt, return_tensors="pt").attention_mask.to(self.model.device)
                else:
                    # Fallback for models without a specific chat template (like base LLMs)
                    # This will concatenate messages as a simple string, which might not be ideal for chat.
                    print(
                        f"Warning: Model {self.model_name} does not have a chat template. Using simple string concatenation.")
                    formatted_string = "\n".join(
                        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in message_list])
                    if self.tokenizer.eos_token_id:
                        formatted_string += self.tokenizer.eos_token  # Add EOS if available
                    inputs = self.tokenizer(
                        formatted_string, return_tensors="pt")
                    inputs = {key: value.to(self.model.device)
                              for key, value in inputs.items()}
                print(f"Input tokens: {inputs['input_ids'].shape}")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                outputs = outputs.to("cpu")
                print(f"Output tokens: {outputs.shape}")
                generated_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True)
                print(f"Generated text: {generated_text}")
                if generated_text.startswith(formatted_string):
                    content = generated_text[len(formatted_string):].strip()
                else:
                    content = generated_text.strip()  # Fallback if prompt isn't at the start

                if content is None or content == "":
                    raise ValueError(
                        "Transformer model returned empty response; retrying"
                    )
                return SamplerResponse(
                    response_text=content,
                    response_metadata={},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
