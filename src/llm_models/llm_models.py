import os
import time
import json
import torch
import boto3
from botocore.config import Config
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    MllamaForCausalLM,
)
from src.utils import base642img, RawInput, IOExamples

AI4CODE_OAI_KEY = "***REMOVED***"
BRACHIO_OAI_KEY = "***REMOVED***"

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
            if "codeinterpreter" in model_name:
                provider = "google-genai"
            else:
                provider = "google"
        elif provider is None:
            provider = "openai"

        assert provider in ["google", "google-genai", "openai", "bedrock", "anthropic"], "Provider must be one of 'google', 'openai', 'bedrock', or 'anthropic'."
        self.provider = provider

        self.model_name = model_name
        if provider in ["google", "openai"]:
            self.client = OpenAI(
                api_key="***REMOVED***" if provider == "google" else AI4CODE_OAI_KEY,
                # api_key="***REMOVED***", #Neelay
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/" if provider == "google" else None
            )
        elif provider == "google-genai":
            self.client = genai.Client(api_key="***REMOVED***")
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
                api_key=AI4CODE_OAI_KEY,
                # api_key="***REMOVED***"
            )

    def _create_completion_with_retry(self, model, messages, max_attempts=5, delay_seconds=2, **kwargs):
        """Calls chat.completions.create with retry logic."""
        response = None
        last_exception = None
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                return response # Return response on success
            except Exception as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    print(f"Encountered {e.__class__.__name__} on attempt {attempt+1}/{max_attempts}. Waiting {delay_seconds} seconds before retrying...")
                    time.sleep(delay_seconds)
                else:
                    print(f"API call failed after {max_attempts} attempts for model {model}.")
                    raise last_exception # Re-raise the exception after the last attempt
        # This part should ideally not be reached if max_attempts > 0
        # but added for completeness in case max_attempts is 0 or negative.
        if last_exception:
            raise last_exception
        return None # Should not happen with positive attempts

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
            response = self._create_completion_with_retry(
                model=self.model_name,
                messages=prompt,
                temperature=sampling_params.temperature,
                max_tokens=10000,
                top_p=1.0,
                n=sampling_params.n,
            )
        elif self.provider == "google-genai":
            # convert the prompt from the openai format to the google genai format
            genai_contents = []
            for message in prompt:
                role = message["role"]
                content = message["content"]

                # Handle different content types
                if isinstance(content, str):
                    # Simple text content
                    genai_contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part(text=content)]
                        )
                    )
                elif isinstance(content, list):
                    # Content with multiple parts (text, images, etc.)
                    parts = []
                    for item in content:
                        if item["type"] == "text":
                            parts.append(types.Part(text=item["text"]))
                        elif item["type"] == "image_url":
                            # Handle image URLs
                            img_url = item["image_url"]["url"]

                            # Check if it's a base64 encoded image
                            if img_url.startswith("data:image"):
                                # Extract the base64 part
                                base64_data = img_url.split(",")[1]

                                # Create an inline data part
                                parts.append(
                                    types.Part(
                                        inline_data=types.Blob(
                                            mime_type="image/jpeg",  # Default to JPEG, adjust if needed
                                            data=base64_data
                                        )
                                    )
                                )
                            else:
                                # It's a regular URL
                                parts.append(
                                    types.Part(
                                        file_data=types.FileData(
                                            file_uri=img_url,
                                            mime_type="image/jpeg"  # Default to JPEG, adjust if needed
                                        )
                                    )
                                )
                    if parts:  # Only add if we have parts
                        genai_contents.append(
                            types.Content(
                                role=role,
                                parts=parts
                            )
                        )

            response = self.client.models.generate_content(
                model=self.model_name.replace("-codeinterpreter", ""),
                contents=genai_contents,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(
                        code_execution=types.ToolCodeExecution
                    )],
                    temperature=sampling_params.temperature,
                    max_output_tokens=10000,
                )
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

            response = self._create_completion_with_retry(
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
        elif self.provider == "google-genai":
            # Extract token counts if available
            if response.usage_metadata is not None:
                print("Prompt tokens:", response.usage_metadata.prompt_token_count)
                print("Response tokens:", response.usage_metadata.candidates_token_count)
            else:
                print("Token counts not available for Google GenAI response")

            # Extract the response text and code execution results
            response_text = ""
            code_execution_results = []

            if response.candidates is not None:
                for candidate in response.candidates:
                    if candidate.content is not None:
                        for part in candidate.content.parts:
                            if part.text is not None:
                                response_text += part.text

                            if part.executable_code is not None:
                                executable_code = part.executable_code
                                if executable_code.code is not None:
                                    code_execution_results.append({
                                        'code': executable_code.code,
                                    })

                            # Check for code execution results
                            if part.code_execution_result is not None:
                                code_result = part.code_execution_result
                                if code_result.output is not None:
                                    code_execution_results.append({
                                        'output': code_result.output,
                                    })

            # Combine text and code execution results
            final_response = response_text
            if code_execution_results:
                for i, result in enumerate(code_execution_results):
                    if "code" in result:
                        final_response += f"Code:\n{result['code']}\n"
                    if "output" in result:
                        final_response += f"Output:\n{result['output']}\n"
            return [Outputs([Text(final_response)])]
        else:
            print("Prompt tokens:", response.usage.prompt_tokens)
            print("Response tokens:", response.usage.completion_tokens)

            return [Outputs([Text(response.choices[i].message.content) for i in range(sampling_params.n)])]