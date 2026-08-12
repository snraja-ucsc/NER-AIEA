"""
models_extra.py

Adds support for:
  1) Local open-weight models served through Ollama (e.g. Qwen2.5, Llama3.1/3.2)
  2) Newer hosted OpenAI chat models (e.g. gpt-4o-mini, gpt-4.1-mini)

Both are accessed through the *same* client class, because Ollama exposes an
OpenAI-compatible /v1/chat/completions endpoint. The only difference between
"local Qwen" and "real GPT-4o" is which base_url/api_key/model string you pass in.

IMPORTANT: This uses the MODERN `openai` python package (>=1.0), which has a
different API surface than the one models.py's original OpenAIGPT class uses
(that class was written against openai==0.28 and will raise AttributeErrors
on a current pip install of `openai`, e.g. `openai.ChatCompletion` and
`openai.error.RateLimitError` no longer exist).

If you need the original OpenAIGPT baseline to run as-is, pin the SDK version
for that specific run:
    pip install "openai==0.28"
then switch to `pip install --upgrade openai` again before using this file.
Trying to use both class styles in the same pip environment does not work,
since they require different package versions.
"""
import os
import time
from openai import OpenAI, RateLimitError, APIError, APIConnectionError


class FlanT5Model:
    """Local FLAN-T5 adapter for PromptNER's existing non-chat code path."""

    def __init__(self, model_name="google/flan-t5-xxl", load_in_8bit=True,
                 max_new_tokens=200, device_map="auto"):
        try:
            import torch
            from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                                      BitsAndBytesConfig)
        except ImportError as exc:
            raise RuntimeError(
                "FLAN-T5 requires torch, transformers, accelerate, and bitsandbytes."
            ) from exc
        self.model = model_name
        self.max_new_tokens = max_new_tokens
        self.seconds_per_query = 0.0
        self.load_in_8bit = load_in_8bit
        if load_in_8bit and not torch.cuda.is_available():
            raise RuntimeError(
                "8-bit FLAN-T5 loading requires CUDA. Run on a GPU Pod, or pass "
                "load_in_8bit=False for a much slower full-precision CPU load."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"device_map": device_map}
        if load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self._torch = torch
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        self._model.eval()

    def is_chat(self):
        return False

    def query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        # With ``device_map=\"auto\"`` this is the device hosting the input
        # embeddings, which is the safe target for the input IDs.
        device = self._model.get_input_embeddings().weight.device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def __call__(self, prompt):
        if not isinstance(prompt, str):
            raise TypeError("FLAN-T5 is a non-chat model and expects a string prompt")
        return self.query(prompt)


def get_flan_t5_model(**kwargs):
    """Construct the paper-control FLAN-T5-XXL model (8-bit by default)."""
    return FlanT5Model(**kwargs)


class ChatModel:
    """
    Generic chat-completion wrapper. Mimics the interface algorithms.py expects
    from models.OpenAIGPT: is_chat(), chat_query(msgs), query(prompt), __call__.

    Works for:
      - Ollama-served local models (base_url="http://localhost:11434/v1", api_key="ollama")
      - Real OpenAI models       (base_url="https://api.openai.com/v1", api_key=<your key>)
    """

    def __init__(self, model, base_url, api_key=None, max_tokens=250,
                 seconds_per_query=0.0, max_retries=5, temperature=0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.seconds_per_query = seconds_per_query
        self.max_retries = max_retries
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "ollama",  # Ollama ignores the key but the client requires a non-empty string
        )

    def is_chat(self):
        # every model this class talks to is a chat/instruct model
        return True

    def request_chat_model(self, msgs):
        messages = [{"role": role, "content": content} for content, role in msgs]
        kwargs = {"model": self.model, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def decode_response(response):
        return response.choices[0].message.content

    def chat_query(self, msgs):
        for attempt in range(self.max_retries):
            try:
                response = self.request_chat_model(msgs)
                return self.decode_response(response)
            except RateLimitError:
                time.sleep(2 ** attempt)
            except (APIError, APIConnectionError) as e:
                # Ollama cold-starts a model on first call; give it a moment and retry
                time.sleep(2)
                if attempt == self.max_retries - 1:
                    raise
        raise RuntimeError(f"Exceeded max_retries ({self.max_retries}) calling {self.model}")

    def query(self, prompt):
        # Fall back path for the non-chat code branch in algorithms.py;
        # we just wrap the plain prompt as a single user turn.
        return self.chat_query([(prompt, "user")])

    def __call__(self, inputs):
        # `inputs` is either a raw string (single-query path) or a list of
        # (content, role) tuples (chat path) depending on how algorithms.py calls us
        if isinstance(inputs, str):
            return self.query(inputs)
        return self.chat_query(inputs)


def get_ollama_model(model_name, base_url="http://localhost:11434/v1", **kwargs):
    """
    Convenience constructor for a local Ollama-served model.
    model_name must match what you pulled, e.g. 'qwen2.5:7b', 'llama3.1:8b'.
    No API key needed; Ollama does not check it.
    """
    return ChatModel(model=model_name, base_url=base_url, api_key="ollama", **kwargs)


def get_openai_model(model_name, **kwargs):
    """
    Convenience constructor for a real, newer OpenAI chat model
    (e.g. 'gpt-4o-mini', 'gpt-4.1-mini'). Reads OPENAI_API_KEY from env.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return ChatModel(model=model_name, base_url="https://api.openai.com/v1",
                      api_key=api_key, seconds_per_query=(60 / 20) + 0.01, **kwargs)
