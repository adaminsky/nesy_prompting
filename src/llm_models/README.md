# LLM Models Module

This module provides wrapper classes for interacting with various Large Language Models (LLMs) in a unified way.

## Classes

### OurLLM

A wrapper for local LLM models, particularly those from Hugging Face like Llama and Qwen.

**Features:**
- Supports Llama-3.2, Llama-3.3, and Qwen models
- Handles both text and image inputs
- Provides consistent interface for model inference

**Usage:**
```python
from src.llm_models import OurLLM

# Initialize the model
model = OurLLM(model_name="meta-llama/Llama-3.2-90B-Vision-Instruct")

# Use the model
response = model.chat(prompt, sampling_params, use_tqdm=False)
```

### APIModel

A wrapper for API-based LLM services like Claude, Gemini, and GPT.

**Features:**
- Supports Claude, Gemini, and GPT models
- Handles API authentication and retries
- Provides consistent interface for model inference

**Usage:**
```python
from src.llm_models import APIModel

# Initialize the model
model = APIModel(model_name="claude-3-opus-20240229")

# Use the model
response = model.chat(prompt, sampling_params, use_tqdm=False)
```

## Common Interface

Both classes implement a common interface with the following methods:

- `__init__(model_name)`: Initialize the model with the specified model name
- `chat(prompt, sampling_params, use_tqdm)`: Generate a response from the model

## Dependencies

- transformers
- torch
- openai
- boto3
- anthropic

## Notes

- API keys should be configured securely, not hardcoded in the source code
- The module is designed to be used with the vLLM framework for local models 