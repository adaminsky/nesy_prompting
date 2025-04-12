import os
import time
import json
import torch
import boto3
from botocore.config import Config
from openai import OpenAI
import anthropic
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    MllamaForCausalLM,
)
from src.utils import base642img, RawInput, IOExamples


class OurLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        if "Llama-3.2" in model_name:
            self.model = MllamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                token="***REMOVED***",
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name, token="***REMOVED***"
            )
        elif "Llama-3.3" in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                token="***REMOVED***",
            )
            self.processor = AutoTokenizer.from_pretrained(
                model_name, token="***REMOVED***"
            )
        elif "Qwen" in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
                token="***REMOVED***",
            )
            self.processor = AutoTokenizer.from_pretrained(
                model_name, token="***REMOVED***"
            )

    def chat(self, prompt, sampling_params, use_tqdm):
        # parse prompt content
        prompt_content = []
        imgs = []
        for i in range(len(prompt[0]["content"])):
            if prompt[0]["content"][i]["type"] == "text":
                if "Qwen" in self.model_name or "Llama-3.3" in self.model_name:
                    prompt_content.append(prompt[0]["content"][i]["text"])
                else:
                    prompt_content.append(prompt[0]["content"][i])
            elif prompt[0]["content"][i]["type"] == "image_url":
                prompt_content.append({"type": "image"})
                img_base64 = prompt[0]["content"][i]["image_url"]["url"].split(",")[1]
                imgs.append(base642img(img_base64))

        if "Qwen" in self.model_name or "Llama-3.3" in self.model_name:
            prompt_content = "".join(prompt_content)

        prompt = [{"role": "user", "content": prompt_content}]

        if "Llama-3.2" in self.model_name:
            input_text = self.processor.apply_chat_template(
                prompt, add_generation_prompt=True
            )
            inputs = self.processor(
                imgs if len(imgs) > 0 else None,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda:0")
        elif "Qwen" in self.model_name or "Llama-3.3" in self.model_name:
            input_text = self.processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(input_text, return_tensors="pt").to("cuda:0")

        print(self.processor.decode(inputs["input_ids"][0]))
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                do_sample=True if sampling_params.temperature > 0 else False,
                top_p=sampling_params.top_p,
                stop_strings=[sampling_params.stop] if sampling_params.stop else None,
            )
        elapsed = time.time() - start

        print(
            f"Tokens per second: {(len(outputs[0][len(inputs['input_ids'][0]):])) / elapsed}"
        )

        output_text = self.processor.decode(outputs[0][len(inputs["input_ids"][0]) :])

        class Outputs:
            def __init__(self, outputs):
                self.outputs = outputs

        class Text:
            def __init__(self, text):
                self.text = text

        return [Outputs([Text(output_text)])]


class APIModel:
    def __init__(self, model_name, provider=None):
        assert "claude" not in model_name or provider is not None, "Provider must be specified for Claude models. Can be 'anthropic' or 'bedrock'."
        if "gemini" in model_name:
            provider = "google"
        elif provider is None:
            provider = "openai"

        assert provider in ["google", "openai", "bedrock", "anthropic"], "Provider must be one of 'google', 'openai', 'bedrock', or 'anthropic'."
        self.provider = provider

        self.model_name = model_name
        if provider in ["google", "openai"]:
            self.client = OpenAI(
                api_key="***REMOVED***" if provider == "google" else "***REMOVED***", #Adam
                # api_key="***REMOVED***", #Neelay
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/" if provider == "google" else None
            )
        elif provider == "bedrock":
            config = Config(
                read_timeout=300,      # Increase read timeout to 300 seconds (adjust as needed)
                connect_timeout=60,    # Optionally increase the connection timeout too
                retries={
                    'max_attempts': 10,  # Also configure retries if desired
                    'mode': 'standard'
                }
            )
            self.client = boto3.client("bedrock-runtime", region_name="us-east-1", config=config)
        elif provider == "anthropic":
            self.client = anthropic.Anthropic(
                api_key="***REMOVED***"
            )
        else:
            self.client = OpenAI(
                api_key="***REMOVED***"
                # api_key="***REMOVED***"
            )

    def chat(self, prompt, sampling_params, use_tqdm):
        if self.provider == "bedrock":
            native_request = {
                "messages": prompt,
                "max_tokens": 131072,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "stop_sequences": [],
                "anthropic_version": "bedrock-2023-05-31"
            }
            request = json.dumps(native_request)
            for attempt in range(4):  # Try up to 4 times
                try:
                    response = self.client.invoke_model(
                        modelId=self.model_name, 
                        body=request,
                        contentType="application/json",
                        accept="application/json"
                    )
                    break  # Exit loop on success
                except Exception as e:
                    if attempt < 3:
                        print(f"Encountered {e.__class__.__name__} on attempt {attempt+1}/4. Waiting 180 seconds before retrying...")
                        time.sleep(180)
                    else:
                        raise e
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                messages=prompt,
                temperature=0.0,
                max_tokens=2500,
                top_p=1.0,
            )
        elif self.provider == "google":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                temperature=sampling_params.temperature,
                max_tokens=10000,
                top_p=1.0,
                n=sampling_params.n,
            )
        else:
            assert self.provider in ["openai"]
            extra_args = {}
            if "o3" in self.model_name:
                extra_args["reasoning_effort"] = "medium"
                extra_args["max_completion_tokens"] = 10000
                extra_args["n"] = sampling_params.n
            else:
                extra_args["max_completion_tokens"] = 10000
                extra_args["n"] = sampling_params.n
                extra_args["temperature"] = sampling_params.temperature
                extra_args["top_p"] = 1.0

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                **extra_args
            )
        class Outputs:
            def __init__(self, outputs):
                self.outputs = outputs

        class Text:
            def __init__(self, text):
                self.text = text

        # print number of tokens
        if self.provider == "bedrock":
            response_body = json.loads(response["body"].read())
            # Claude 3 has a 'usage' field with input/output token counts
            print("Prompt tokens:", response_body['usage']['input_tokens'])
            print("Response tokens:", response_body['usage']['output_tokens'])
            response_text = response_body["content"][0]["text"]
            return [Outputs([Text(response_text)])]
        elif self.provider == "anthropic":
            print("Prompt tokens:", response.usage.input_tokens)
            print("Response tokens:", response.usage.output_tokens)
            response = response.content[0].text
            return [Outputs([Text(response)])]
        else:
            print("Prompt tokens:", response.usage.prompt_tokens)
            print("Response tokens:", response.usage.completion_tokens)
        
        return [Outputs([Text(response.choices[i].message.content) for i in range(sampling_params.n)])] 